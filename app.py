import io
import math
import time

import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity


def quadtree_compress(image: np.ndarray, threshold: float, min_block: int,
                      lossless: bool = False):
    """
    Adaptive pixel-value splitting.
    - Calculate per-channel standard deviation in each region.
    - If variation is high, split the region into four.
    - If variation is low, represent it by its mean colour (lossy mode).
    - At minimum block size, retain original pixels to protect fine details.
    - In lossless mode, retain every original pixel (splitting is diagnostic only).
    """
    h, w, channels = image.shape
    reconstructed = image.copy()
    leaf_count = 0
    split_count = 0
    preserved_pixels = 0
    approximated_pixels = 0

    def process(y0, y1, x0, x1):
        nonlocal leaf_count, split_count, preserved_pixels, approximated_pixels
        block = image[y0:y1, x0:x1]
        bh, bw = y1 - y0, x1 - x0
        leaf_count += 1

        # A region's variation is the largest standard deviation across channels.
        variation = float(np.max(np.std(block.astype(np.float32), axis=(0, 1))))

        if lossless:
            preserved_pixels += bh * bw
            return

        can_split = (bh > min_block or bw > min_block) and (bh > 1 or bw > 1)
        if variation > threshold and can_split:
            split_count += 1
            ym = y0 + bh // 2
            xm = x0 + bw // 2
            # Split only valid, non-empty quadrants (also handles odd dimensions).
            regions = [
                (y0, ym, x0, xm),
                (y0, ym, xm, x1),
                (ym, y1, x0, xm),
                (ym, y1, xm, x1),
            ]
            for a, b, c, d in regions:
                if a < b and c < d:
                    process(a, b, c, d)
        elif variation <= threshold:
            # Low-variation block: use its mean colour.
            mean_colour = np.mean(block.astype(np.float32), axis=(0, 1))
            reconstructed[y0:y1, x0:x1] = np.clip(
                np.rint(mean_colour), 0, 255
            ).astype(np.uint8)
            approximated_pixels += bh * bw
        else:
            # High-detail block reached the minimum size: preserve it exactly.
            preserved_pixels += bh * bw

    process(0, h, 0, w)
    stats = {
        "leaf_regions": leaf_count,
        "split_operations": split_count,
        "preserved_pixels": preserved_pixels,
        "approximated_pixels": approximated_pixels,
    }
    return reconstructed, stats


def png_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray):
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)

    ssim = float(structural_similarity(
        original, reconstructed, channel_axis=2, data_range=255
    ))
    return mse, psnr, ssim


st.set_page_config(page_title="Pixel-Based Image Compression", page_icon="🖼️", layout="wide")
st.title("Minimum-Loss Image Compression")
st.subheader("Pixel Value-Based Adaptive Splitting for Survival Applications")
st.write(
    "This prototype uses adaptive quadtree splitting. Regions with high pixel "
    "variation are divided; low-variation regions can be represented by their "
    "mean colour. Fine-detail regions are preserved at the minimum block size."
)

with st.sidebar:
    st.header("Compression settings")
    mode = st.radio(
        "Mode",
        ["Safety-first (lossless)", "Adaptive minimum-loss (lossy)"],
        help="Use lossless mode for medical, rescue, or other critical images."
    )
    threshold = st.slider(
        "Pixel variation threshold", min_value=0, max_value=80,
        value=12, step=1,
        help="Lower values cause more splitting and usually preserve more detail."
    )
    min_block = st.select_slider(
        "Minimum block size (pixels)",
        options=[2, 4, 8, 16, 32],
        value=4
    )
    st.caption("The threshold is the maximum per-channel standard deviation allowed before splitting.")

uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded:
    try:
        source = Image.open(uploaded).convert("RGB")
        original = np.asarray(source, dtype=np.uint8)
        is_lossless = mode.startswith("Safety-first")

        started = time.perf_counter()
        reconstructed, stats = quadtree_compress(
            original, threshold, min_block, lossless=is_lossless
        )
        elapsed = time.perf_counter() - started

        original_png = png_bytes(original)
        compressed_png = png_bytes(reconstructed)
        mse, psnr, ssim = calculate_metrics(original, reconstructed)
        ratio = len(original_png) / max(1, len(compressed_png))
        reduction = (1 - len(compressed_png) / max(1, len(original_png))) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original image")
            st.image(original, use_container_width=True)
            st.caption(f"PNG baseline: {len(original_png):,} bytes")
        with col2:
            st.markdown("#### Reconstructed image")
            st.image(reconstructed, use_container_width=True)
            st.caption(f"Reconstructed PNG: {len(compressed_png):,} bytes")

        st.markdown("### Performance")
        a, b, c, d = st.columns(4)
        a.metric("MSE", f"{mse:.4f}")
        b.metric("PSNR", "∞ (identical)" if math.isinf(psnr) else f"{psnr:.2f} dB")
        c.metric("SSIM", f"{ssim:.5f}")
        d.metric("Size reduction", f"{reduction:.2f}%")

        st.write(f"**Compression ratio (PNG baseline / output):** {ratio:.3f}:1")
        st.write(f"**Processing time:** {elapsed * 1000:.2f} ms")
        st.write(f"**Quadtree split operations:** {stats['split_operations']}")
        st.write(f"**Leaf regions:** {stats['leaf_regions']}")
        st.write(f"**Pixels approximated:** {stats['approximated_pixels']:,}")
        st.write(f"**Pixels preserved exactly:** {stats['preserved_pixels']:,}")

        if is_lossless:
            st.success(
                "Safety-first mode: every pixel is preserved exactly. "
                "This guarantees zero reconstruction loss, but does not guarantee "
                "a smaller file than the input."
            )
        else:
            st.warning(
                "Adaptive mode is lossy: low-variation regions are replaced by "
                "their mean colour. Do not use this mode as the only copy of "
                "medical, navigation, or rescue-critical imagery."
            )

        st.download_button(
            "Download reconstructed image (PNG)",
            data=compressed_png,
            file_name="reconstructed_image.png",
            mime="image/png"
        )

        st.download_button(
            "Download pixel difference map (PNG)",
            data=png_bytes(np.abs(
                original.astype(np.int16) - reconstructed.astype(np.int16)
            ).astype(np.uint8)),
            file_name="difference_map.png",
            mime="image/png"
        )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload an image to begin.")
