"""构造"思考模式"（CoT）SFT 数据：可验证的算术/多步应用题 + 少量通用指令。

输出格式（与 minigpt/model/generation.py 的协议一致）：
    instruction: 题目
    input:       ""（InstructionDataset 约定字段）
    output:      "让我逐步分析：\n1. …\n2. …\n最终答案：<数字>"

用法：
    python scripts/build_cot_sft.py --out-train dataset/sft/sft_cot_zh.jsonl \
        --out-eval dataset/sft/cot_eval_zh.jsonl --n-train 30000 --n-eval 200
"""
import argparse
import json
import os
import random

THINK = "让我逐步分析：\n"
MARK = "最终答案："


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
def gen_easy_add(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return _fmt(f"请计算 {a} + {b} 等于多少？",
                [f"先看个位：{a} + {b}", f"相加得 {a + b}"], a + b)


def gen_easy_sub(rng):
    a, b = rng.randint(2, 9), rng.randint(1, 9)
    a, b = max(a, b), min(a, b)
    return _fmt(f"请计算 {a} - {b} 等于多少？",
                [f"被减数 {a}，减数 {b}", f"相减得 {a - b}"], a - b)


def gen_easy_mul(rng):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    return _fmt(f"请计算 {a} × {b} 等于多少？",
                [f"{a} 个 {b} 相加", f"{a} × {b} = {a * b}"], a * b)


def gen_easy_div(rng):
    b, q = rng.randint(2, 9), rng.randint(2, 9)
    a = b * q
    return _fmt(f"请计算 {a} ÷ {b} 等于多少？",
                [f"想 {b} 乘几等于 {a}", f"{b} × {q} = {a}，所以商是 {q}"], q)


def gen_easy_two_step(rng):
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    return _fmt(f"请计算 {a} + {b} - {c} 等于多少？",
                [f"先算加法：{a} + {b} = {a + b}", f"再算减法：{a + b} - {c} = {a + b - c}"],
                a + b - c)


def gen_easy_word(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
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
    ap.add_argument("--out-eval", default="dataset/sft/cot_eval_zh.jsonl")
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--mix-general", type=float, default=0.25,
                    help="混入通用指令的比例（保持闲聊/问答能力，避免灾难性遗忘）")
    ap.add_argument("--general-jsonl", default="dataset/sft/sft_data_zh.jsonl")
    ap.add_argument("--profile", choices=["hard", "easy"], default="hard",
                    help="hard=多位数/应用题（容量要求高）；easy=1~2 位数单步/两步（小模型可学）")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    n_general = int(args.n_train * args.mix_general)
    builder = build_easy if args.profile == "easy" else build
    math_rows = builder(args.n_train - n_general, seed=args.seed)
    general = load_general(args.general_jsonl, n_general, seed=args.seed + 1)
    train = math_rows + general
    random.Random(args.seed).shuffle(train)
    write_jsonl(train, args.out_train)

    # 评测集用不同 seed 与生成器偏移，避免与训练题目重合
    gen_list = EASY_GENERATORS if args.profile == "easy" else GENERATORS
    eval_rows = builder(args.n_eval, seed=args.seed + 99999, offset=len(gen_list) // 2)
    write_jsonl(eval_rows, args.out_eval)


if __name__ == "__main__":
    main()
