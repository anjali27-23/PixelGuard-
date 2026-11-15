# Minimum-Loss Image Compression Using Pixel Value-Based Splitting

## What this prototype does
- Reads an image and converts it to RGB.
- Measures pixel variation in each region.
- Splits high-variation regions into four (adaptive quadtree decomposition).
- In adaptive mode, represents low-variation regions by their mean colour.
- Preserves high-detail regions when the minimum block size is reached.
- Reports MSE, PSNR, SSIM, processing time, and PNG size comparison.

## Important safety note
Use **Safety-first (lossless)** mode for medical, rescue, navigation, or other critical imagery. It preserves every pixel exactly. The quadtree analysis is then diagnostic only; the output may not be smaller than the input. The adaptive mode is lossy and is for experimentation only, not as the sole copy of critical data.

## Run on Windows
1. Install Python 3.10 or newer.
2. Open Command Prompt in this folder.
3. Run:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL shown by Streamlit and upload an image.

## Metrics
- MSE: mean squared pixel error (lower is better).
- PSNR: peak signal-to-noise ratio in dB (higher is better).
- SSIM: structural similarity (closer to 1 is better).
- Size reduction: compares optimized PNG output against an optimized PNG baseline made from the uploaded image.

## Research limitation
This is a working academic prototype, not a validated survival-grade codec. PNG file-size changes depend on image content and may be negative. For deployment, add region-of-interest protection, authenticated metadata, corruption detection, and validation against clinically/operationally approved source images.


## Deploy to Render.com

### Option A: Deploy using the Render Blueprint
1. Upload this project folder to a GitHub repository.
2. Sign in to Render and choose **New +** → **Blueprint**.
3. Connect the GitHub repository containing `render.yaml`.
4. Render reads `render.yaml` and creates the web service.
5. Wait for the build and deployment to finish.
6. Open the `onrender.com` URL shown on the service page.

### Option B: Create a Web Service manually
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Health check path: `/_stcore/health`

### Before publishing
- Test with ordinary images and large images.
- Use Safety-first (lossless) mode for critical imagery.
- Do not treat this academic prototype as a certified medical or rescue system.
- Render free services may sleep when idle and can take time to wake up.
