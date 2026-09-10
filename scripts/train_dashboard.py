"""训练看板：解析预训练日志 + checkpoint + GPU，输出网页看板 / 终端文本 / 损失走势图 PNG。

用法：
  1) 浏览器看板（每 5 秒刷新，默认端口 8099）：
       python3 scripts/train_dashboard.py --serve --port 8099
     打开 http://127.0.0.1:8099
     页面含：进度条、train/eval loss 走势图（带坐标轴/网格/悬停数值）、grad_norm 走势图、
            速率与 ETA、GPU、checkpoint 列表、日志尾部。
  2) 终端看板（含 ASCII 走势图）：
       bash scripts/watch_training.sh          # 每 5s 刷新
  3) 导出损失走势图 PNG（用于报告/README）：
       python3 scripts/train_dashboard.py --plot models/checkpoints/loss_curve.png
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EVAL_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+lr=(?P<lr>[\d.eE+-]+),\s+"
    r"train_loss:\s*(?P<train>[\d.]+),\s*eval_loss:\s*(?P<eval>[\d.]+),\s*"
    r"grad_norm=(?P<grad>[\d.]+),\s*steps:\s*(?P<step>\d+)/(?P<total>\d+)")
START_RE = re.compile(r"start epoch:(?P<epoch>\d+) from step:(?P<step>\d+)")
RESTART_RE = re.compile(r"第 (?P<n>\d+) 次启动")
SPARK = "▁▂▃▄▅▆▇█"


def parse_log(path, tail_bytes=8_000_000):
    if not os.path.exists(path):
        return {"evals": [], "epoch": None, "restarts": 0, "size": 0}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - tail_bytes))
        text = f.read().decode("utf-8", "replace")
    evals, epoch, restarts = [], None, 0
    for line in text.splitlines():
        m = EVAL_RE.search(line)
        if m:
            d = m.groupdict()
            evals.append({"ts": d["ts"], "lr": float(d["lr"]), "train": float(d["train"]),
                          "eval": float(d["eval"]), "grad": float(d["grad"]),
                          "step": int(d["step"]), "total": int(d["total"])})
            continue
        m = START_RE.search(line)
        if m:
            epoch = int(m.group("epoch"))
            continue
        m = RESTART_RE.search(line)
        if m:
            restarts = max(restarts, int(m.group("n")))
    return {"evals": evals[-500:], "epoch": epoch, "restarts": restarts, "size": size}


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        return {"util": int(util), "mem_used": int(used), "mem_total": int(total), "temp": int(temp)}
    except Exception:  # noqa: BLE001
        return {}


def checkpoints(out_dir):
    rows = []
    for f in glob.glob(os.path.join(out_dir, "*.pth")):
        st = os.stat(f)
        rows.append({"name": os.path.basename(f), "size_mb": round(st.st_size / 1e6, 1),
                     "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S"),
                     "atime": st.st_mtime})
    rows.sort(key=lambda r: r["atime"])
    return [{k: v for k, v in r.items() if k != "atime"} for r in rows[-8:]]


def resolve_run(args, scan_bytes=2_000_000):
    """auto 模式：在 models/checkpoints/*.log 中挑选最近仍在产出评估的日志（支持 SFT 阶段）。"""
    import glob as _glob
    if args.log != "auto":
        log = args.log
        out = args.out_dir if args.out_dir != "auto" else os.path.splitext(log)[0]
        return log, out, os.path.basename(log)
    candidates = []
    for pat in ("models/checkpoints/*.log", "*.log"):
        candidates += _glob.glob(pat)
    best, best_ts = None, ""
    for path in set(candidates):
        if path.endswith((".watchdog.log", ".downstream.log")):
            continue
        info = parse_log(path, tail_bytes=scan_bytes)
        if info["evals"]:
            ts = info["evals"][-1]["ts"]
            if ts > best_ts:
                best, best_ts = path, ts
    if best is None:
        best = args.log if args.log != "auto" else "models/checkpoints/pretrain_v2_full.log"
    out = os.path.splitext(best)[0]
    return best, out, os.path.basename(best)


def snapshot(args):
    args.log, auto_out, run_name = resolve_run(args)
    if args.out_dir == "auto":
        args.out_dir = auto_out
    data = parse_log(args.log)
    wd_log = args.watchdog_log or (args.out_dir.rstrip("/") + ".watchdog.log")
    wd = parse_log(wd_log) if os.path.exists(wd_log) else {"restarts": 0}
    data["restarts"] = max(data.get("restarts", 0), wd.get("restarts", 0))
    evals = data["evals"]
    last = evals[-1] if evals else None
    rate = None
    if len(evals) >= 2:
        a, b = evals[-2], evals[-1]
        t0 = time.mktime(time.strptime(a["ts"], "%Y-%m-%d %H:%M:%S"))
        t1 = time.mktime(time.strptime(b["ts"], "%Y-%m-%d %H:%M:%S"))
        dsteps, dt = b["step"] - a["step"], max(t1 - t0, 1e-6)
        if dsteps > 0:
            rate = {"s_per_step": dt / dsteps, "tok_per_s": dsteps * args.tokens_per_step / dt}
    step = last["step"] if last else 0
    target = args.target_step
    if last and last.get("total", 0) > step:
        target = last["total"]          # 用当前 run 自己的总步数（预训练 422520 / SFT 14700 …）
    eta_min = (target - step) * rate["s_per_step"] / 60 if rate and target > step else None
    prog = min(1.0, step / target) if target else 0.0
    try:
        with open(args.log, "rb") as f:
            f.seek(max(0, os.path.getsize(args.log) - 20000))
            tail = f.read().decode("utf-8", "replace").splitlines()[-12:]
    except OSError:
        tail = []
    return {"now": datetime.now().strftime("%F %T"), "run": run_name, "step": step, "target_step": target,
            "progress": round(prog, 4), "epoch": data["epoch"], "restarts": data["restarts"],
            "log_size_mb": round(data["size"] / 1e6, 1), "last": last, "rate": rate,
            "eta_min": None if eta_min is None else round(eta_min, 1), "evals": evals[-300:],
            "checkpoints": checkpoints(args.out_dir), "gpu": gpu_info(), "tail": tail,
            "final_exists": os.path.exists(os.path.join(args.out_dir, "final.pt")),
            "best_exists": os.path.exists(os.path.join(args.out_dir, "best.pt"))}


def ascii_spark(values, width=64):
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return SPARK[0] * len(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in vals)


def render_text(s):
    lines = [f"[{s['now']}]  训练看板（每 5s 刷新）  run={s.get('run','-')}",
             f"进度: {s['step']}/{s['target_step']}  ({s['progress']*100:.1f}%)  epoch={s['epoch']}  重启={s['restarts']}"]
    if s["last"]:
        l = s["last"]
        lines.append(f"最新: train {l['train']:.4f} | eval {l['eval']:.4f} | lr {l['lr']:.2e} | grad {l['grad']:.3f}")
    if s["evals"]:
        lines.append(f"eval损失走势(最近{min(len(s['evals']),64)}点): {ascii_spark([e['eval'] for e in s['evals']])}  ↓越低越好")
    if s["rate"]:
        lines.append(f"速率: {s['rate']['s_per_step']:.3f}s/step  {s['rate']['tok_per_s']/1000:.1f}k tok/s"
                     + (f"  ETA {s['eta_min']/60:.1f}h" if s["eta_min"] else ""))
    if s["gpu"]:
        g = s["gpu"]
        lines.append(f"GPU: util {g['util']}% | mem {g['mem_used']}/{g['mem_total']}MB | temp {g['temp']}C")
    cps = ", ".join(f"{c['name']}({c['mtime']})" for c in s["checkpoints"][-3:])
    lines.append(f"checkpoints: {cps or '—'}")
    if s["best_exists"]:
        lines.append("⭐ best.pt 已产出（eval_loss 历史最优，下游/评测优先用它）")
    if s["final_exists"]:
        lines.append("✅ final.pt 已产出（预训练完成）")
    return "\n".join(lines)


def plot_loss(s, path):
    """导出 train/eval loss（+ lr）走势图 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = s["evals"]
    if not ev:
        raise SystemExit("没有可绘制的评估点")
    steps = [e["step"] for e in ev]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, [e["train"] for e in ev], label="train_loss", lw=1.6)
    ax.plot(steps, [e["eval"] for e in ev], label="eval_loss", lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.set_title(f"MiniGPT pretraining loss  (step {s['step']}/{s['target_step']})")
    ax2 = ax.twinx()
    ax2.plot(steps, [e["lr"] for e in ev], color="tab:green", ls="--", lw=1.0, label="lr")
    ax2.set_ylabel("lr", color="tab:green")
    ax2.tick_params(axis="y", colors="tab:green")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[dashboard] loss curve saved -> {path}")


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>MiniGPT 训练看板</title>
<style>
 body{background:#0f1115;color:#e6e6e6;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}
 h1{font-size:18px;margin:0 0 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
 .card{background:#171a21;border:1px solid #232833;border-radius:10px;padding:14px}
 .k{color:#8b93a7;font-size:12px}
 .v{font-size:22px;font-weight:600;margin-top:2px}
 .bar{height:12px;background:#232833;border-radius:6px;overflow:hidden;margin:10px 0}
 .bar>i{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#22c55e)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{padding:3px 6px;border-bottom:1px solid #232833;text-align:left}
 pre{white-space:pre-wrap;color:#9aa4b2;font-size:12px;max-height:200px;overflow:auto}
 svg.chart{width:100%;height:260px;display:block}
 .legend{font-size:12px;color:#9aa4b2;margin-top:6px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
</style></head><body>
<h1>MiniGPT 训练看板 <span id="now" class="k"></span></h1>
<div class="grid">
 <div class="card"><div class="k">进度（step / target）</div><div class="v" id="prog">-</div>
   <div class="bar"><i id="bar" style="width:0%"></i></div><div class="k" id="meta"></div></div>
 <div class="card"><div class="k">train / eval loss</div><div class="v" id="loss">-</div><div class="k" id="lr"></div></div>
 <div class="card"><div class="k">速率 / ETA</div><div class="v" id="rate">-</div><div class="k" id="eta"></div></div>
 <div class="card"><div class="k">GPU</div><div class="v" id="gpu">-</div><div class="k" id="gpumem"></div></div>
</div>
<div class="card" style="margin-top:14px">
  <div class="k">loss 走势（train / eval，x=step）</div>
  <svg class="chart" id="lossChart" viewBox="0 0 900 260" preserveAspectRatio="none"></svg>
  <div class="legend" id="lossLegend"></div>
</div>
<div class="grid" style="margin-top:14px">
  <div class="card"><div class="k">grad_norm 走势</div>
    <svg class="chart" id="gradChart" viewBox="0 0 900 200" preserveAspectRatio="none"></svg>
    <div class="legend" id="gradLegend"></div></div>
  <div class="card"><div class="k">最近 checkpoint</div><table id="ckpt"></table></div>
</div>
<div class="card" style="margin-top:14px"><div class="k">日志尾部</div><pre id="tail"></pre></div>
<script>
const REFRESH = __REFRESH__;
const GRAD_CLIP = __GRAD_CLIP__;
function chart(svg, series, opts){
  const W=900,H=opts.h,mL=62,mR=14,mT=14,mB=28, iw=W-mL-mR, ih=H-mT-mB;
  let pts=series.flatMap(s=>s.points);
  if(pts.length<2){svg.innerHTML='<text x="20" y="30" fill="#8b93a7">等待数据…</text>';return;}
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  if(y1-y0<1e-6){y1=y0+1e-6;}
  const pad=(y1-y0)*0.08; y0-=pad; y1+=pad;
  const sx=v=>mL+(v-x0)/((x1-x0)||1)*iw, sy=v=>mT+(1-(v-y0)/((y1-y0)||1))*ih;
  let g='';
  for(let i=0;i<=4;i++){const y=mT+ih*i/4, val=y1-(y1-y0)*i/4;
    g+=`<line x1="${mL}" y1="${y.toFixed(1)}" x2="${W-mR}" y2="${y.toFixed(1)}" stroke="#232833"/>`;
    g+=`<text x="${mL-6}" y="${(y+4).toFixed(1)}" fill="#8b93a7" font-size="11" text-anchor="end">${val.toFixed(opts.digits)}</text>`;}
  for(let i=0;i<=5;i++){const x=mL+iw*i/5, val=x0+(x1-x0)*i/5;
    g+=`<line x1="${x.toFixed(1)}" y1="${mT}" x2="${x.toFixed(1)}" y2="${mT+ih}" stroke="#1c2129"/>`;
    g+=`<text x="${x.toFixed(1)}" y="${H-8}" fill="#8b93a7" font-size="11" text-anchor="middle">${(val/1000).toFixed(0)}k</text>`;}
  if(opts.refLine!==undefined && opts.refLine>=y0 && opts.refLine<=y1){
    const yr=sy(opts.refLine);
    g+=`<line x1="${mL}" y1="${yr.toFixed(1)}" x2="${W-mR}" y2="${yr.toFixed(1)}" stroke="#ef4444" stroke-dasharray="6 4" stroke-width="1.2"/>`;
    g+=`<text x="${W-mR-4}" y="${(yr-5).toFixed(1)}" fill="#ef4444" font-size="11" text-anchor="end">clip ${opts.refLine}</text>`;
  }
  let body='';
  for(const s of series){
    const d=s.points.map(p=>`${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
    body+=`<polyline fill="none" stroke="${s.color}" stroke-width="1.8" points="${d}"/>`;
    const step=Math.max(1,Math.floor(s.points.length/120));
    s.points.filter((_,i)=>i%step===0).forEach(p=>{
      body+=`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.2" fill="${s.color}"><title>${s.name}\nstep ${p.x}\n${p.y.toFixed(4)}\n${p.ts||''}</title></circle>`;});
    const last=s.points[s.points.length-1];
    body+=`<text x="${(sx(last.x)+6).toFixed(1)}" y="${(sy(last.y)-6).toFixed(1)}" fill="${s.color}" font-size="11">${last.y.toFixed(opts.digits)}</text>`;
  }
  const legend=series.map(s=>`<span class="sw" style="background:${s.color}"></span>${s.name}`).join('');
  svg.innerHTML=g+body;
  const lg=document.getElementById(opts.legendId);
  if(lg) lg.innerHTML=legend;
}
async function tick(){
 try{
  const d=await (await fetch('/data')).json();
  document.getElementById('now').textContent=d.now;
  document.getElementById('prog').textContent=`${d.step} / ${d.target_step} (${(d.progress*100).toFixed(1)}%)`;
  document.getElementById('bar').style.width=(d.progress*100).toFixed(1)+'%';
  document.getElementById('meta').textContent=`epoch ${d.epoch} · 重启 ${d.restarts} 次 · 日志 ${d.log_size_mb}MB`+(d.final_exists?" · ✅ 已完成":"");
  if(d.last){document.getElementById('loss').textContent=`${d.last.train.toFixed(4)} / ${d.last.eval.toFixed(4)}`;
    document.getElementById('lr').textContent=`lr ${d.last.lr.toExponential(2)} · grad_norm ${d.last.grad.toFixed(3)}`;}
  if(d.rate){document.getElementById('rate').textContent=`${d.rate.s_per_step.toFixed(3)} s/step · ${(d.rate.tok_per_s/1000).toFixed(1)}k tok/s`;
    document.getElementById('eta').textContent=d.eta_min?`ETA ${(d.eta_min/60).toFixed(1)} 小时`:'—';}
  if(d.gpu.util!==undefined){document.getElementById('gpu').textContent=`${d.gpu.util}% · ${d.gpu.temp}C`;
    document.getElementById('gpumem').textContent=`显存 ${d.gpu.mem_used}/${d.gpu.mem_total} MB`;}
  const ev=d.evals||[];
  chart(document.getElementById('lossChart'),
        [{name:'train_loss',color:'#3b82f6',points:ev.map(e=>({x:e.step,y:e.train,ts:e.ts}))},
         {name:'eval_loss', color:'#22c55e',points:ev.map(e=>({x:e.step,y:e.eval, ts:e.ts}))}],
        {h:260,digits:3,legendId:'lossLegend'});
  const gv=ev.map(e=>e.grad);
  const gstats=gv.length?`  min ${Math.min(...gv).toFixed(3)} · mean ${(gv.reduce((a,b)=>a+b,0)/gv.length).toFixed(3)} · max ${Math.max(...gv).toFixed(3)}`:'';
  chart(document.getElementById('gradChart'),
        [{name:'grad_norm',color:'#f59e0b',points:ev.map(e=>({x:e.step,y:e.grad,ts:e.ts}))}],
        {h:200,digits:2,legendId:'gradLegend',refLine:GRAD_CLIP});
  const gl=document.getElementById('gradLegend'); if(gl) gl.textContent=(gl.textContent||'')+gstats;
  document.getElementById('ckpt').innerHTML=d.checkpoints.slice().reverse().map(c=>
    `<tr><td>${c.name}</td><td>${c.size_mb}MB</td><td>${c.mtime}</td></tr>`).join('');
  document.getElementById('tail').textContent=d.tail.join('\n');
 }catch(e){document.getElementById('now').textContent='看板连接失败: '+e;}
}
tick(); setInterval(tick, REFRESH*1000);
</script></body></html>
"""


def make_handler(args):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/data"):
                body = json.dumps(snapshot(args), ensure_ascii=False).encode()
                ctype = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body = (PAGE.replace("__REFRESH__", str(args.refresh))
                    .replace("__GRAD_CLIP__", str(args.grad_clip))).encode()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="auto",
                    help="训练日志；默认 auto=自动选取最近仍有评估输出的 run 日志")
    ap.add_argument("--out-dir", default="auto",
                    help="checkpoint 目录；默认 auto=与日志同名的 run 目录")
    ap.add_argument("--target-step", type=int, default=211000)
    ap.add_argument("--tokens-per-step", type=int, default=8 * 511)
    ap.add_argument("--refresh", type=int, default=5)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--plot", default=None, help="导出损失走势图 PNG 后退出")
    ap.add_argument("--watchdog-log", default=None,
                    help="自愈守护日志（默认 <out-dir>.watchdog.log），用于统计重启次数")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="梯度裁剪阈值：在看板 grad 图上画参考线（与 trainer 的 grad_clip 一致）")
    args = ap.parse_args()

    if args.plot:
        plot_loss(snapshot(args), args.plot)
    elif args.serve:
        srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(args))
        print(f"[dashboard] serving at http://127.0.0.1:{args.port} (refresh {args.refresh}s)", flush=True)
        srv.serve_forever()
    else:
        print(render_text(snapshot(args)))


if __name__ == "__main__":
    main()
