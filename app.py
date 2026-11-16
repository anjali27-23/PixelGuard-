"""Flask web UI for the Pixel-Value-Splitting (PVS) codec."""
import base64
import io
import os

import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image

import pvs_codec as pvs

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap
MAX_SIDE = int(os.environ.get("MAX_SIDE", 160))

BAND_COLORS = np.array([[37, 99, 235], [16, 185, 129], [245, 158, 11]], dtype=np.uint8)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PVS Codec | Minimum-Loss Image Compression</title>
<style>
  :root{
    --bg:#070b1a; --panel:rgba(255,255,255,.06); --panel-b:rgba(255,255,255,.12);
    --text:#e8ecf8; --muted:#9aa6c4; --accent:#6366f1; --accent2:#22d3ee;
    --ok:#34d399; --bad:#f87171;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    margin:0; min-height:100vh; color:var(--text);
    font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:
      radial-gradient(900px 500px at 12% -10%, rgba(99,102,241,.40), transparent 60%),
      radial-gradient(800px 500px at 95% 5%, rgba(34,211,238,.25), transparent 60%),
      radial-gradient(900px 600px at 50% 115%, rgba(168,85,247,.28), transparent 60%),
      linear-gradient(180deg,#070b1a,#0b1226 60%,#070b1a);
    background-attachment:fixed;
  }
  body::before{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
      linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size:44px 44px;
    -webkit-mask-image:radial-gradient(ellipse at 50% 30%, #000 30%, transparent 75%);
            mask-image:radial-gradient(ellipse at 50% 30%, #000 30%, transparent 75%);
  }
  .wrap{position:relative; z-index:1; max-width:1120px; margin:0 auto; padding:0 1.25rem 3rem}
  nav{display:flex; align-items:center; justify-content:space-between; padding:1.25rem 0}
  .brand{display:flex; align-items:center; gap:.7rem; font-weight:700; letter-spacing:.2px}
  .logo{
    width:34px; height:34px; border-radius:9px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    display:grid; grid-template-columns:repeat(3,1fr); gap:2px; padding:6px;
  }
  .logo i{background:rgba(255,255,255,.9); border-radius:1.5px; display:block}
  .logo i:nth-child(2n){opacity:.55}
  .tag{font-size:.75rem; color:var(--muted); border:1px solid var(--panel-b); padding:.25rem .65rem; border-radius:999px}
  header.hero{text-align:center; padding:2.2rem 0 1.6rem}
  .hero h1{
    margin:0 0 .6rem; font-size:clamp(1.9rem,4.5vw,3rem); line-height:1.1; letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#a5b4fc 45%,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .hero p{margin:0 auto; max-width:680px; color:var(--muted); line-height:1.6}
  .steps{display:flex; justify-content:center; flex-wrap:wrap; gap:.5rem; margin-top:1.4rem}
  .step{font-size:.78rem; padding:.35rem .8rem; border-radius:999px; background:var(--panel); border:1px solid var(--panel-b); color:#c7d2fe}
  .step b{color:var(--accent2); margin-right:.35rem}
  .grid{display:grid; grid-template-columns:360px 1fr; gap:1.25rem; align-items:start}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  .card{
    background:var(--panel); border:1px solid var(--panel-b); border-radius:18px; padding:1.4rem;
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    box-shadow:0 10px 40px rgba(0,0,0,.35);
  }
  .card h2{margin:0 0 1rem; font-size:1rem; letter-spacing:.02em; display:flex; align-items:center; gap:.5rem}
  .card h2::before{content:""; width:4px; height:16px; border-radius:2px; background:linear-gradient(var(--accent),var(--accent2))}
  label.f{display:block; font-size:.78rem; color:var(--muted); margin:1.1rem 0 .4rem; text-transform:uppercase; letter-spacing:.06em}
  .drop{
    display:block; border:1.5px dashed rgba(165,180,252,.45); border-radius:14px; padding:1.4rem 1rem;
    text-align:center; cursor:pointer; transition:.2s; background:rgba(99,102,241,.07);
  }
  .drop:hover{border-color:var(--accent2); background:rgba(34,211,238,.08)}
  .drop svg{width:30px; height:30px; stroke:#a5b4fc}
  .drop span{display:block; margin-top:.4rem; font-size:.85rem; color:var(--muted)}
  .drop strong{color:var(--text)}
  .drop input{display:none}
  .switch{display:flex; align-items:center; gap:.6rem; font-size:.9rem; cursor:pointer}
  .switch input{accent-color:var(--accent); width:17px; height:17px}
  .row{display:grid; gap:.6rem}
  .row.two{grid-template-columns:1fr 1fr}
  .row.three{grid-template-columns:repeat(3,1fr)}
  .field small{display:block; color:var(--muted); font-size:.7rem; margin-bottom:.25rem}
  input[type=number]{
    width:100%; padding:.6rem .7rem; border-radius:10px; border:1px solid var(--panel-b);
    background:rgba(8,12,28,.6); color:var(--text); font-size:.95rem; outline:none;
  }
  input[type=number]:focus{border-color:var(--accent); box-shadow:0 0 0 3px rgba(99,102,241,.25)}
  input[type=number]:disabled{opacity:.4}
  .hint{font-size:.75rem; color:var(--muted); margin-top:.5rem; line-height:1.45}
  button.go{
    margin-top:1.4rem; width:100%; padding:.85rem 1rem; border:0; border-radius:12px; cursor:pointer;
    font-size:1rem; font-weight:600; color:#fff; letter-spacing:.02em;
    background:linear-gradient(135deg,var(--accent),#8b5cf6 55%,var(--accent2));
    box-shadow:0 8px 24px rgba(99,102,241,.4); transition:transform .15s, box-shadow .15s;
  }
  button.go:hover{transform:translateY(-1px); box-shadow:0 12px 30px rgba(99,102,241,.55)}
  button.go:disabled{opacity:.7; cursor:progress; transform:none}
  .err{margin-top:1rem; padding:.8rem 1rem; border-radius:10px; background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.4); color:#fecaca; font-size:.88rem}
  .empty{text-align:center; padding:3.5rem 1rem; color:var(--muted)}
  .empty svg{width:54px; height:54px; stroke:#6366f1; opacity:.7; margin-bottom:.6rem}
  .metrics{display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem}
  @media(max-width:700px){.metrics{grid-template-columns:repeat(2,1fr)}}
  .metric{padding:.9rem; border-radius:14px; background:rgba(8,12,28,.55); border:1px solid var(--panel-b)}
  .metric small{display:block; color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.07em}
  .metric div{font-size:1.45rem; font-weight:700; margin-top:.25rem; background:linear-gradient(90deg,#fff,#a5b4fc); -webkit-background-clip:text; background-clip:text; color:transparent}
  .metric em{font-style:normal; font-size:.75rem; color:var(--muted); -webkit-text-fill-color:var(--muted)}
  .chips{display:flex; flex-wrap:wrap; gap:.5rem; margin:1.1rem 0 .2rem; font-size:.8rem}
  .chip{padding:.3rem .7rem; border-radius:999px; background:rgba(8,12,28,.55); border:1px solid var(--panel-b); color:#c7d2fe}
  .chip.ok{color:var(--ok); border-color:rgba(52,211,153,.4)} .chip.bad{color:var(--bad); border-color:rgba(248,113,113,.4)}
  .bars{margin:1.2rem 0 .4rem}
  .bar{display:flex; align-items:center; gap:.8rem; font-size:.8rem; color:var(--muted); margin:.45rem 0}
  .bar span:first-child{width:90px}
  .track{flex:1; height:10px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden}
  .fill{height:100%; border-radius:6px}
  .fill.a{background:linear-gradient(90deg,var(--accent),var(--accent2))}
  .fill.b{background:rgba(255,255,255,.35)}
  .gallery{display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-top:1.2rem}
  @media(max-width:760px){.gallery{grid-template-columns:repeat(2,1fr)}}
  .gallery figure{margin:0}
  .gallery img{width:100%; display:block; border-radius:10px; border:1px solid var(--panel-b); image-rendering:pixelated; background:#000}
  .gallery figcaption{font-size:.75rem; color:var(--muted); text-align:center; margin-top:.4rem}
  .legend{display:flex; gap:.9rem; flex-wrap:wrap; font-size:.72rem; color:var(--muted); margin-top:.7rem}
  .legend i{display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:.35rem; vertical-align:-1px}
  a.dl{
    display:inline-flex; align-items:center; gap:.5rem; margin-top:1.2rem; padding:.6rem 1rem;
    border-radius:10px; text-decoration:none; color:var(--text); font-size:.88rem;
    border:1px solid var(--panel-b); background:rgba(255,255,255,.06);
  }
  a.dl:hover{border-color:var(--accent2)}
  footer{margin-top:2.5rem; text-align:center; color:var(--muted); font-size:.78rem}
  .overlay{position:fixed; inset:0; z-index:10; display:none; align-items:center; justify-content:center; flex-direction:column; gap:1rem;
    background:rgba(7,11,26,.78); backdrop-filter:blur(6px); color:var(--text)}
  .overlay.on{display:flex}
  .spin{width:46px; height:46px; border-radius:50%; border:4px solid rgba(255,255,255,.15); border-top-color:var(--accent2); animation:sp 0.9s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="overlay" id="ov"><div class="spin"></div><div>Compressing and verifying round trip&hellip;</div></div>
<div class="wrap">
  <nav>
    <div class="brand"><div class="logo"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>PVS Codec</div>
    <div class="tag">Near-lossless &middot; Arithmetic coded</div>
  </nav>

  <header class="hero">
    <h1>Minimum-loss image compression</h1>
    <p>Pixels are split into intensity bands. Critical bands stay lossless, the rest get a
    guaranteed error bound, and everything is entropy coded with context-adaptive arithmetic coding.</p>
    <div class="steps">
      <span class="step"><b>1</b>Split by pixel value</span>
      <span class="step"><b>2</b>Band map</span>
      <span class="step"><b>3</b>Bounded offsets</span>
      <span class="step"><b>4</b>Arithmetic coding</span>
    </div>
  </header>

  <div class="grid">
    <form class="card" method="post" enctype="multipart/form-data" id="frm">
      <h2>Settings</h2>
      <label class="drop" id="drop">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
        <span id="fname"><strong>Choose an image</strong> or drop it here</span>
        <input type="file" name="image" id="file" accept="image/*" required>
      </label>

      <label class="f">Band thresholds</label>
      <label class="switch"><input type="checkbox" name="otsu" id="otsu" {{ 'checked' if form.otsu }}> Auto-detect (multi-Otsu, 3 bands)</label>
      <div class="row two" style="margin-top:.7rem">
        <div class="field"><small>Threshold 1</small><input type="number" name="t1" id="t1" min="1" max="254" value="{{ form.t1 }}"></div>
        <div class="field"><small>Threshold 2</small><input type="number" name="t2" id="t2" min="2" max="255" value="{{ form.t2 }}"></div>
      </div>

      <label class="f">Error bound &delta; per band</label>
      <div class="row three">
        <div class="field"><small>Dark</small><input type="number" name="d0" min="0" max="8" value="{{ form.d0 }}"></div>
        <div class="field"><small>Mid</small><input type="number" name="d1" min="0" max="8" value="{{ form.d1 }}"></div>
        <div class="field"><small>Bright</small><input type="number" name="d2" min="0" max="8" value="{{ form.d2 }}"></div>
      </div>
      <div class="hint">&delta; = 0 makes a band lossless. Larger values compress more but allow up to &plusmn;&delta; error per pixel.</div>

      <button class="go" type="submit" id="btn">Compress &amp; verify</button>
      {% if error %}<div class="err">{{ error }}</div>{% endif %}
      <div class="hint" style="margin-top:1rem">Images are converted to grayscale and downscaled to {{ max_side }}px (longest side).</div>
    </form>

    <section class="card">
      <h2>Results</h2>
      {% if r %}
        <div class="metrics">
          <div class="metric"><small>Compression</small><div>{{ '%.2f'|format(r.cr) }}:1</div></div>
          <div class="metric"><small>PSNR</small><div>{{ r.psnr_val }}{% if r.psnr_unit %} <em>{{ r.psnr_unit }}</em>{% endif %}</div></div>
          <div class="metric"><small>Bits / pixel</small><div>{{ '%.2f'|format(r.bpp) }}</div></div>
          <div class="metric"><small>SSIM</small><div>{{ '%.4f'|format(r.ssim) }}</div></div>
        </div>

        <div class="chips">
          <span class="chip">Thresholds {{ r.thresholds[0] }} / {{ r.thresholds[1] }}</span>
          <span class="chip">&delta; {{ r.deltas[0] }} / {{ r.deltas[1] }} / {{ r.deltas[2] }}</span>
          <span class="chip">Max error {{ r.maxerr }}</span>
          <span class="chip {{ 'ok' if r.bound else 'bad' }}">{{ 'Error bound respected' if r.bound else 'Bound violated' }}</span>
        </div>

        <div class="bars">
          <div class="bar"><span>Raw</span><div class="track"><div class="fill b" style="width:100%"></div></div><span>{{ r.raw }} B</span></div>
          <div class="bar"><span>PNG</span><div class="track"><div class="fill b" style="width:{{ r.png_pct }}%"></div></div><span>{{ r.png }} B</span></div>
          <div class="bar"><span>PVS</span><div class="track"><div class="fill a" style="width:{{ r.pvs_pct }}%"></div></div><span>{{ r.size }} B</span></div>
        </div>

        <div class="gallery">
          <figure><img src="data:image/png;base64,{{ r.orig }}" alt="Original"><figcaption>Original</figcaption></figure>
          <figure><img src="data:image/png;base64,{{ r.dec }}" alt="Decoded"><figcaption>Decoded</figcaption></figure>
          <figure><img src="data:image/png;base64,{{ r.bandmap }}" alt="Band map"><figcaption>Band map</figcaption></figure>
          <figure><img src="data:image/png;base64,{{ r.errmap }}" alt="Error map"><figcaption>Error map ({{ 'amplified' if r.maxerr else 'none' }})</figcaption></figure>
        </div>
        <div class="legend">
          <span><i style="background:#2563eb"></i>Dark band</span>
          <span><i style="background:#10b981"></i>Mid band</span>
          <span><i style="background:#f59e0b"></i>Bright band</span>
        </div>

        <a class="dl" download="image.pvs" href="data:application/octet-stream;base64,{{ r.blob }}">&#8681; Download compressed .pvs</a>
      {% else %}
        <div class="empty">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.7"/><path d="m21 16-5-5-8 8"/></svg>
          <div>Upload an image to see compression metrics,<br>the band map and an error heatmap.</div>
        </div>
      {% endif %}
    </section>
  </div>

  <footer>PVS Codec &middot; pixel-value-based splitting with bounded-error coding</footer>
</div>

<script>
  const file = document.getElementById('file'), fname = document.getElementById('fname');
  const drop = document.getElementById('drop');
  file.addEventListener('change', () => { if (file.files[0]) fname.innerHTML = '<strong>' + file.files[0].name + '</strong>'; });
  ['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); }));
  drop.addEventListener('drop', ev => { ev.preventDefault(); if (ev.dataTransfer.files.length) { file.files = ev.dataTransfer.files; file.dispatchEvent(new Event('change')); } });
  const otsu = document.getElementById('otsu'), t1 = document.getElementById('t1'), t2 = document.getElementById('t2');
  const sync = () => { t1.disabled = t2.disabled = otsu.checked; };
  otsu.addEventListener('change', sync); sync();
  document.getElementById('frm').addEventListener('submit', () => {
    t1.disabled = t2.disabled = false;
    document.getElementById('ov').classList.add('on');
    document.getElementById('btn').disabled = true;
  });
</script>
</body>
</html>
"""


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def error_heatmap(err):
    m = int(err.max())
    if m == 0:
        return np.zeros(err.shape + (3,), dtype=np.uint8)
    v = err.astype(np.float64) / m
    rgb = np.stack([30 + 225 * v, 20 + 140 * v, 60 + 20 * (1 - v)], axis=-1)
    return rgb.astype(np.uint8)


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def index():
    form = dict(otsu=True, t1=64, t2=192, d0=2, d1=0, d2=2)
    result, error = None, None

    if request.method == "POST":
        try:
            form = dict(
                otsu="otsu" in request.form,
                t1=int(request.form["t1"]), t2=int(request.form["t2"]),
                d0=int(request.form["d0"]), d1=int(request.form["d1"]), d2=int(request.form["d2"]),
            )
            f = request.files.get("image")
            if not f or not f.filename:
                raise ValueError("Please choose an image.")
            try:
                im = Image.open(f.stream).convert("L")
            except Exception:
                raise ValueError("That file could not be read as an image.")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            img = np.array(im, dtype=np.uint8)

            if form["otsu"]:
                th = pvs.multi_otsu_3(img)
            else:
                th = (form["t1"], form["t2"])
                if not (1 <= th[0] < th[1] <= 255):
                    raise ValueError("Thresholds must satisfy 1 <= t1 < t2 <= 255.")
            deltas = (form["d0"], form["d1"], form["d2"])

            blob = pvs.encode(img, th, deltas)
            rec = pvs.decode(blob)

            bands = np.digitize(img, th)
            err = np.abs(img.astype(int) - rec.astype(int))
            allowed = np.array(deltas)[bands]
            p = pvs.psnr(img, rec)
            pbuf = io.BytesIO()
            Image.fromarray(img).save(pbuf, format="PNG", optimize=True)
            png_size, raw = pbuf.tell(), int(img.size)
            top = max(raw, len(blob), png_size)

            result = Obj(
                thresholds=tuple(int(t) for t in th), deltas=deltas, size=len(blob),
                bpp=8 * len(blob) / img.size, cr=img.size / len(blob),
                psnr_val="Lossless" if p == float("inf") else f"{p:.1f}",
                psnr_unit="" if p == float("inf") else "dB",
                ssim=pvs.ssim_global(img, rec), maxerr=int(err.max()),
                bound=bool(np.all(err <= allowed)), png=png_size, raw=raw,
                png_pct=round(100 * png_size / top, 1), pvs_pct=round(100 * len(blob) / top, 1),
                orig=png_b64(img), dec=png_b64(rec),
                bandmap=png_b64(BAND_COLORS[bands]), errmap=png_b64(error_heatmap(err)),
                blob=base64.b64encode(blob).decode(),
            )
        except Exception as e:
            error = str(e)

    return render_template_string(PAGE, form=Obj(**form), r=result, error=error, max_side=MAX_SIDE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
