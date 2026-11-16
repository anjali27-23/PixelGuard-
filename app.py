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

# The codec is pure Python, so cap the working resolution to keep requests fast
# on small hosting tiers.
MAX_SIDE = int(os.environ.get("MAX_SIDE", 160))

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PVS Image Codec</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
  h1{font-size:1.5rem} fieldset{border:1px solid #ccc;border-radius:8px;margin:1rem 0;padding:1rem}
  label{display:block;margin:.5rem 0 .2rem;font-size:.9rem}
  input[type=number]{width:5rem} button{padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
  table{border-collapse:collapse;margin:1rem 0} td,th{border:1px solid #ccc;padding:.4rem .8rem;text-align:left}
  .imgs{display:flex;gap:1rem;flex-wrap:wrap} .imgs figure{margin:0}
  img{image-rendering:pixelated;border:1px solid #ccc;max-width:100%}
  .err{color:#b00020} .ok{color:#0a7d2c}
</style>
</head>
<body>
<h1>Pixel-Value-Splitting image compression</h1>
<p>Splits pixels into intensity bands, keeps critical bands lossless and applies a
bounded error (&delta;) to the others. Images are downscaled to at most {{ max_side }} px
on the longest side and converted to grayscale.</p>

<form method="post" enctype="multipart/form-data">
  <fieldset>
    <label>Image file</label>
    <input type="file" name="image" accept="image/*" required>
    <label><input type="checkbox" name="otsu" {{ 'checked' if form.otsu }}> Auto thresholds (multi-Otsu, 3 bands)</label>
    <label>Manual thresholds (used if Otsu is off)</label>
    <input type="number" name="t1" min="1" max="254" value="{{ form.t1 }}">
    <input type="number" name="t2" min="2" max="255" value="{{ form.t2 }}">
    <label>Error bound &delta; per band (0 = lossless): low / mid / high</label>
    <input type="number" name="d0" min="0" max="8" value="{{ form.d0 }}">
    <input type="number" name="d1" min="0" max="8" value="{{ form.d1 }}">
    <input type="number" name="d2" min="0" max="8" value="{{ form.d2 }}">
  </fieldset>
  <button type="submit">Compress &amp; decode</button>
</form>

{% if error %}<p class="err">{{ error }}</p>{% endif %}

{% if r %}
<h2>Results</h2>
<table>
  <tr><th>Thresholds</th><td>{{ r.thresholds }}</td></tr>
  <tr><th>&delta; per band</th><td>{{ r.deltas }}</td></tr>
  <tr><th>Size</th><td>{{ r.size }} bytes ({{ '%.3f'|format(r.bpp) }} bpp)</td></tr>
  <tr><th>Compression ratio</th><td>{{ '%.2f'|format(r.cr) }} : 1</td></tr>
  <tr><th>PSNR</th><td>{{ r.psnr }}</td></tr>
  <tr><th>SSIM (global)</th><td>{{ '%.4f'|format(r.ssim) }}</td></tr>
  <tr><th>Max |error|</th><td>{{ r.maxerr }}</td></tr>
  <tr><th>Error bound respected</th><td class="{{ 'ok' if r.bound else 'err' }}">{{ r.bound }}</td></tr>
  <tr><th>PNG baseline</th><td>{{ r.png }} bytes</td></tr>
</table>
<div class="imgs">
  <figure><img src="data:image/png;base64,{{ r.orig }}" alt="original"><figcaption>Original</figcaption></figure>
  <figure><img src="data:image/png;base64,{{ r.dec }}" alt="decoded"><figcaption>Decoded</figcaption></figure>
</div>
<p><a download="image.pvs" href="data:application/octet-stream;base64,{{ r.blob }}">Download compressed .pvs file</a></p>
{% endif %}
</body>
</html>
"""


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def index():
    form = {"otsu": True, "t1": 64, "t2": 192, "d0": 2, "d1": 0, "d2": 2}
    result, error = None, None

    if request.method == "POST":
        try:
            form = {
                "otsu": "otsu" in request.form,
                "t1": int(request.form["t1"]),
                "t2": int(request.form["t2"]),
                "d0": int(request.form["d0"]),
                "d1": int(request.form["d1"]),
                "d2": int(request.form["d2"]),
            }
            f = request.files.get("image")
            if not f or not f.filename:
                raise ValueError("Please choose an image.")
            im = Image.open(f.stream).convert("L")
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

            err = np.abs(img.astype(int) - rec.astype(int))
            allowed = np.array(deltas)[np.digitize(img, th)]
            p = pvs.psnr(img, rec)
            pbuf = io.BytesIO()
            Image.fromarray(img).save(pbuf, format="PNG", optimize=True)

            result = {
                "thresholds": tuple(int(t) for t in th),
                "deltas": deltas,
                "size": len(blob),
                "bpp": 8 * len(blob) / img.size,
                "cr": img.size / len(blob),
                "psnr": "lossless" if p == float("inf") else f"{p:.2f} dB",
                "ssim": pvs.ssim_global(img, rec),
                "maxerr": int(err.max()),
                "bound": bool(np.all(err <= allowed)),
                "png": pbuf.tell(),
                "orig": png_b64(img),
                "dec": png_b64(rec),
                "blob": base64.b64encode(blob).decode(),
            }
        except Exception as e:  # show a friendly message instead of a 500
            error = str(e)

    return render_template_string(PAGE, form=type("F", (), form), r=type("R", (), result) if result else None,
                                  error=error, max_side=MAX_SIDE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
