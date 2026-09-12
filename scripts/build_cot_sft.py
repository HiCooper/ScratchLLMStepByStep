"""构造"思考模式"（CoT）SFT 数据：可验证的算术/多步应用题 + 少量通用指令。

输出格式（与 minigpt/model/generation.py 的协议一致）：
    instruction: 题目
    input:       ""（InstructionDataset 约定字段）
    output:      "让我逐步分析：\n1. …\n2. …\n最终答案：<数字>"

用法：
    python scripts/build_cot_sft.py --out-train dataset/sft/sft_cot_zh.jsonl \
        --out-eval dataset/sft/cot_eval_hard_disjoint.jsonl --n-train 30000 --n-eval 200
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 格式标记的**唯一**定义在 minigpt/cot_format.py（推理与评测也从那里取）——
# 以前这里和 generation.py 各写一份，改一处就会静默让评测掉进"取最后一个数字"的兜底。
from minigpt.cot_format import ANSWER_MARK, THINK_PREFIX  # noqa: E402

THINK = THINK_PREFIX
MARK = ANSWER_MARK


def _fmt(question, steps, answer):
    body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return {"instruction": question, "input": "", "output": f"{THINK}{body}\n{MARK}{answer}"}


def gen_addition(rng):
    a, b = rng.randint(100, 9999), rng.randint(100, 9999)
    return _fmt(f"请计算 {a} + {b} 等于多少？",
                [f"把两个数相加：{a} + {b}", f"个位到万位逐位相加得 {a + b}"], a + b)


def gen_subtraction(rng):
    a, b = rng.randint(1000, 9999), rng.randint(100, 999)
    a, b = max(a, b), min(a, b)
    return _fmt(f"请计算 {a} - {b} 等于多少？",
                [f"被减数 {a}，减数 {b}", f"逐位相减得 {a - b}"], a - b)


def gen_multiplication(rng):
    a, b = rng.randint(12, 99), rng.randint(3, 19)
    return _fmt(f"请计算 {a} × {b} 等于多少？",
                [f"把 {b} 拆成 {b // 10 * 10} + {b % 10}",
                 f"{a} × {b // 10 * 10} = {a * (b // 10 * 10)}，{a} × {b % 10} = {a * (b % 10)}",
                 f"两部分相加：{a * (b // 10 * 10)} + {a * (b % 10)}"], a * b)


def gen_mixed(rng):
    a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
    return _fmt(f"请计算 {a} + {b} - {c} 等于多少？",
                [f"先算加法：{a} + {b} = {a + b}", f"再算减法：{a + b} - {c} = {a + b - c}"], a + b - c)


def gen_word_apples(rng):
    a, b, c = rng.randint(10, 60), rng.randint(5, 30), rng.randint(1, 20)
    return _fmt(f"小明原来有 {a} 个苹果，又买了 {b} 个，然后吃掉了 {c} 个。请问现在还剩多少个苹果？",
                [f"原有的苹果：{a} 个", f"买了以后共有：{a} + {b} = {a + b} 个",
                 f"吃掉后剩下：{a + b} - {c} = {a + b - c} 个"], a + b - c)


def gen_average(rng):
    n = rng.randint(3, 6)
    nums = [rng.randint(60, 100) for _ in range(n)]
    total = sum(nums)
    avg = total / n
    kw = "平均分" if rng.random() < 0.5 else "平均数"
    return _fmt(f"某组数据为 {', '.join(map(str, nums))}，请问它们的{kw}是多少（保留一位小数即可）？",
                [f"先求总和：{' + '.join(map(str, nums))} = {total}",
                 f"共有 {n} 个数据，总和除以个数：{total} ÷ {n} = {avg:.1f}"], f"{avg:.1f}")


def gen_price(rng):
    price, qty = rng.randint(5, 99), rng.randint(2, 30)
    return _fmt(f"一本笔记本 {price} 元，买 {qty} 本需要多少钱？",
                [f"单价 {price} 元，数量 {qty} 本",
                 f"总价 = 单价 × 数量 = {price} × {qty} = {price * qty} 元"], price * qty)


def gen_discount(rng):
    price = rng.randint(100, 999)
    off = rng.choice([10, 15, 20, 25, 30])
    final = round(price * (100 - off) / 100, 1)
    return _fmt(f"一件商品原价 {price} 元，现在打 {100 - off} 折，需要付多少钱？",
                [f"折扣为 {100 - off} 折，即支付原价的 {(100 - off)}%",
                 f"应付 = {price} × {(100 - off)}% = {final} 元"], final)


def gen_three_step(rng):
    a, b, c = rng.randint(100, 500), rng.randint(10, 90), rng.randint(2, 9)
    return _fmt(f"请计算 ({a} + {b}) × {c} 等于多少？",
                [f"先算括号内：{a} + {b} = {a + b}",
                 f"再做乘法：{a + b} × {c} = {(a + b) * c}"], (a + b) * c)


GENERATORS = [gen_addition, gen_subtraction, gen_multiplication, gen_mixed,
              gen_word_apples, gen_average, gen_price, gen_discount, gen_three_step]


# ------------------------- easy profile（容量匹配：1~2 位数、答案短） -------------------------
# easy 的题目空间由操作数上界决定：默认 1..9 时全部唯一题目只有约 1000 个，**不足以**
# 切出与训练集不相交的评测集（实测 1064 个唯一题目全部出现在 60k 训练集里，所以
# "easy CoT 90%→100%" 是记忆而非泛化）。需要做可信评测时用 --easy-max 放大空间。
EASY_MAX = 9


def gen_easy_add(rng):
    a, b = rng.randint(1, EASY_MAX), rng.randint(1, EASY_MAX)
    return _fmt(f"请计算 {a} + {b} 等于多少？",
                [f"先看个位：{a} + {b}", f"相加得 {a + b}"], a + b)


def gen_easy_sub(rng):
    a, b = rng.randint(2, EASY_MAX), rng.randint(1, EASY_MAX)
    a, b = max(a, b), min(a, b)
    return _fmt(f"请计算 {a} - {b} 等于多少？",
                [f"被减数 {a}，减数 {b}", f"相减得 {a - b}"], a - b)


def gen_easy_mul(rng):
    a, b = rng.randint(2, EASY_MAX), rng.randint(2, EASY_MAX)
    return _fmt(f"请计算 {a} × {b} 等于多少？",
                [f"{a} 个 {b} 相加", f"{a} × {b} = {a * b}"], a * b)


def gen_easy_div(rng):
    b, q = rng.randint(2, EASY_MAX), rng.randint(2, EASY_MAX)
    a = b * q
    return _fmt(f"请计算 {a} ÷ {b} 等于多少？",
                [f"想 {b} 乘几等于 {a}", f"{b} × {q} = {a}，所以商是 {q}"], q)


def gen_easy_two_step(rng):
    a, b, c = rng.randint(1, EASY_MAX), rng.randint(1, EASY_MAX), rng.randint(1, EASY_MAX)
    return _fmt(f"请计算 {a} + {b} - {c} 等于多少？",
                [f"先算加法：{a} + {b} = {a + b}", f"再算减法：{a + b} - {c} = {a + b - c}"],
                a + b - c)


def gen_easy_word(rng):
    a, b = rng.randint(1, EASY_MAX), rng.randint(1, EASY_MAX)
    return _fmt(f"小明有 {a} 支铅笔，又买了 {b} 支，一共有多少支铅笔？",
                [f"原有的铅笔：{a} 支", f"又买了 {b} 支", f"一共 {a} + {b} = {a + b} 支"], a + b)


EASY_GENERATORS = [gen_easy_add, gen_easy_sub, gen_easy_mul, gen_easy_div,
                   gen_easy_two_step, gen_easy_word]


def build_easy(n, seed, offset=0):
    rng = random.Random(seed)
    return [EASY_GENERATORS[(i + offset) % len(EASY_GENERATORS)](rng) for i in range(n)]


def build(n, seed, offset=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gen = GENERATORS[(i + offset) % len(GENERATORS)]
        rows.append(gen(rng))
    return rows


def generate_unique_pool(generators, seed, target, exclude=()):
    """按题目文本去重地生成题目池，并排除 `exclude` 中的题目。

    真实问题：旧实现"评测集换个 seed"来避免与训练集重合，但 easy profile 的
    题目空间极小——个位数加减乘除的全部唯一题目只有约 1000 个，200 道评测题
    100% 落在训练集里（实测 easy 训练集仅 1064 个唯一题目，评测集 200/200 命中）。
    于是 README 的"easy CoT 90%→100%"测的是记忆，不是泛化。
    这里改为先构造去重题目池，再切分，从根本上保证 train/eval 不相交。
    """
    rng = random.Random(seed)
    seen = {q.strip() for q in exclude}
    pool, attempts = [], 0
    limit = max(2000, target * 300)
    while len(pool) < target and attempts < limit:
        q = generators[attempts % len(generators)](rng)
        attempts += 1
        key = q["instruction"]
        if key in seen:
            continue
        seen.add(key)
        pool.append(q)
    rng.shuffle(pool)
    return pool, len(seen)


def build_disjoint(n_math, n_eval, seed, profile, exclude=()):
    """返回 (train_math, eval_rows, unique_total)，保证评测题不出现在训练题中。"""
    gens = EASY_GENERATORS if profile == "easy" else GENERATORS
    pool, unique_total = generate_unique_pool(gens, seed, n_math + n_eval, exclude=exclude)
    if len(pool) < n_math + n_eval:
        print(f"[cot-data] 警告：{profile} profile 去重后只有 {len(pool)} 个可用题目"
              f"（含排除项共 {unique_total}），少于请求的 {n_math + n_eval}。"
              f"评测集将缩小以保证与训练集不相交。")
    n_eval_eff = min(n_eval, max(1, len(pool) // 5), max(1, len(pool) - 1))
    eval_rows = pool[:n_eval_eff]
    rest = pool[n_eval_eff:]
    if not rest:
        raise SystemExit(f"[cot-data] {profile} 题目空间不足以切出不相交的评测集")
    # 训练题不足时循环复制（只重复训练题，绝不引入评测题）
    train_math = [rest[i % len(rest)] for i in range(n_math)] if n_math > 0 else []
    print(f"[cot-data] profile={profile} 唯一题目={len(pool)} "
          f"train={len(train_math)}(去重后 {len(rest)}) eval={len(eval_rows)}(与训练集不相交)")
    return train_math, eval_rows, len(pool)


def load_general(path, n, seed):
    if not path or not os.path.exists(path) or n <= 0:
        return []
    rng = random.Random(seed)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({"instruction": d.get("instruction", ""), "input": d.get("input", ""),
                         "output": d.get("output", "")})
    rng.shuffle(rows)
    return rows[:n]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[cot-data] {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="dataset/sft/sft_cot_zh.jsonl")
    ap.add_argument("--out-eval", default="dataset/sft/cot_eval_hard_disjoint.jsonl")
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--mix-general", type=float, default=0.25,
                    help="混入通用指令的比例（保持闲聊/问答能力，避免灾难性遗忘）")
    ap.add_argument("--general-jsonl", default="dataset/sft/sft_data_zh.jsonl")
    ap.add_argument("--profile", choices=["hard", "easy"], default="hard",
                    help="hard=多位数/应用题（容量要求高）；easy=1~2 位数单步/两步（小模型可学）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exclude-jsonl", action="append", default=[],
                    help="构建评测集时排除这些文件里出现过的题面（可多次传入）。"
                         "用于生成与**既有训练集**不相交的评测集，例如排除 "
                         "dataset/sft/sft_cot_easy_em50_60k.jsonl")
    ap.add_argument("--easy-max", type=int, default=9,
                    help="easy profile 的操作数上界（默认 9）。设大（如 20）可放大题目空间，"
                         "从而切出与训练集真正不相交的评测集；改变它会使新旧 easy 结果不可比")
    ap.add_argument("--eval-only", action="store_true",
                    help="只生成评测集（不写训练集），配合 --exclude-jsonl 得到干净的留出集")
    args = ap.parse_args()

    global EASY_MAX
    EASY_MAX = int(args.easy_max)

    exclude = []
    for p in args.exclude_jsonl:
        if not os.path.exists(p):
            print(f"[cot-data] 警告：--exclude-jsonl {p} 不存在，忽略")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    exclude.append(json.loads(line).get("instruction", ""))
    if exclude:
        print(f"[cot-data] 从 {len(args.exclude_jsonl)} 个文件载入 {len(exclude)} 条待排除题面")

    n_general = int(args.n_train * args.mix_general)
    n_math = max(0, args.n_train - n_general)
    math_rows, eval_rows, n_unique = build_disjoint(
        n_math, args.n_eval, args.seed, args.profile, exclude=exclude)
    write_jsonl(eval_rows, args.out_eval)
    if args.eval_only:
        print(f"[cot-data] eval-only：未写训练集（唯一题目池 {n_unique}）")
        return

    general = load_general(args.general_jsonl, n_general, seed=args.seed + 1)
    if n_general > 0 and not general:
        # 真实隐患：通用语料缺失时旧实现静默返回 []，训练集只有数学题却毫无提示
        print(f"[cot-data] 警告：--general-jsonl {args.general_jsonl} 不存在或为空，"
              f"本次训练集将不含通用指令（丢失 {n_general} 条，可能加剧灾难性遗忘）")
    train = math_rows + general
    random.Random(args.seed).shuffle(train)
    write_jsonl(train, args.out_train)

    # 自检：评测题绝不能出现在训练集里
    train_q = {r["instruction"] for r in train}
    overlap = sum(1 for r in eval_rows if r["instruction"] in train_q)
    if overlap:
        raise SystemExit(f"[cot-data] 严重错误：评测集有 {overlap}/{len(eval_rows)} 条与训练集重合")
    print(f"[cot-data] 自检通过：eval({len(eval_rows)}) 与 train({len(train)}) 题面零重合")



if __name__ == "__main__":
    main()
