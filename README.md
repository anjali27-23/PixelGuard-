# PVS Image Codec

Minimum-loss image compression using pixel-value-based splitting.
Pixels are split into intensity bands; critical bands are coded losslessly and the
rest are near-lossless with a guaranteed error bound (delta). Band map and offsets
are coded with context-adaptive arithmetic coding.

## Files
- `pvs_codec.py` – codec (encoder, decoder, metrics) and CLI
- `app.py` – Flask web UI
- `render.yaml` – Render blueprint

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

CLI: `python pvs_codec.py image.png --otsu --deltas 2 0 2`

## Deploy on Render
1. Push this repo to GitHub.
2. Render dashboard -> New -> Blueprint (or Web Service) -> select the repo.
3. Blueprint reads `render.yaml`. For a manual Web Service use:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
   - Health check path: `/health`

Note: the codec is pure Python, so uploads are downscaled to `MAX_SIDE` px (default 160).
