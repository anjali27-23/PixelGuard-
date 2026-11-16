"""
Pixel-Value-Splitting (PVS) minimum-loss image codec
=====================================================
Pipeline
  1. Split the grey-level range into bands (fixed thresholds or multi-Otsu).
  2. Band map      : label of every pixel, coded with a context-adaptive
                     arithmetic coder (context = left + top labels).
  3. Offset stream : offset of the pixel inside its band.
        - critical band     (delta = 0) -> lossless
        - non-critical band (delta > 0) -> near-lossless, |error| <= delta
     coded with a per-band, neighbour-conditioned adaptive arithmetic coder.
  4. Decoder rebuilds map, then offsets, then merges them.

Guarantee: for every pixel, |original - decoded| <= delta of its band.

Usage
  python pvs_codec.py                     # synthetic demo
  python pvs_codec.py image.png           # your own image (needs Pillow)
  python pvs_codec.py image.png --deltas 2 0 2 --otsu
"""
import struct
import sys
import zlib
import argparse
import numpy as np

# --------------------------------------------------------------------------
# Adaptive arithmetic coder (Witten-Neal-Cleary, 32-bit)
# --------------------------------------------------------------------------
CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = HALF >> 1
THREE_Q = HALF + QUARTER
MAX_TOTAL = 1 << 16
INC = 24


class Model:
    """Adaptive frequency model over n symbols."""

    def __init__(self, n):
        self.n = n
        self.freq = [1] * n
        self.total = n

    def find(self, target):
        cum = 0
        for s, f in enumerate(self.freq):
            if cum + f > target:
                return s, cum
            cum += f
        raise ValueError("bad target")

    def cum_of(self, s):
        return sum(self.freq[:s])

    def update(self, s):
        self.freq[s] += INC
        self.total += INC
        if self.total > MAX_TOTAL:
            self.freq = [(f + 1) >> 1 for f in self.freq]
            self.total = sum(self.freq)


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, TOP, 0
        self.bits = []

    def _out(self, b):
        self.bits.append(b)
        if self.pending:
            self.bits.extend([1 - b] * self.pending)
            self.pending = 0

    def encode(self, model, s):
        lo = model.cum_of(s)
        hi = lo + model.freq[s]
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi // model.total - 1
        self.low = self.low + rng * lo // model.total
        while True:
            if self.high < HALF:
                self._out(0)
            elif self.low >= HALF:
                self._out(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_Q:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
        model.update(s)

    def finish(self):
        self.pending += 1
        self._out(0 if self.low < QUARTER else 1)
        bits = self.bits + [0] * ((-len(self.bits)) % 8)
        arr = np.packbits(np.array(bits, dtype=np.uint8))
        return arr.tobytes()


class Decoder:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0
        self.low, self.high = 0, TOP
        self.value = 0
        for _ in range(CODE_BITS):
            self.value = (self.value << 1) | self._bit()

    def _bit(self):
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
        else:
            b = 0
        self.pos += 1
        return b

    def decode(self, model):
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * model.total - 1) // rng
        s, lo = model.find(target)
        hi = lo + model.freq[s]
        self.high = self.low + rng * hi // model.total - 1
        self.low = self.low + rng * lo // model.total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_Q:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._bit()
        model.update(s)
        return s


# --------------------------------------------------------------------------
# Band splitting helpers
# --------------------------------------------------------------------------
def multi_otsu_3(img):
    """Two thresholds (3 bands) maximising between-class variance."""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    best, bt = -1.0, (85, 170)
    for t1 in range(1, 254):
        for t2 in range(t1 + 1, 256):
            w0, w1, w2 = w[t1 - 1], w[t2 - 1] - w[t1 - 1], 1 - w[t2 - 1]
            if min(w0, w1, w2) < 1e-9:
                continue
            m0 = m[t1 - 1] / w0
            m1 = (m[t2 - 1] - m[t1 - 1]) / w1
            m2 = (mt - m[t2 - 1]) / w2
            var = w0 * (m0 - mt) ** 2 + w1 * (m1 - mt) ** 2 + w2 * (m2 - mt) ** 2
            if var > best:
                best, bt = var, (t1, t2)
    return bt


def band_edges(thresholds):
    """Return list of (lo, hi) inclusive for every band."""
    edges = [0, *thresholds, 256]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def quant_params(lo, hi, delta):
    width = hi - lo + 1
    step = 2 * delta + 1
    nsym = (width - 1 + delta) // step + 1
    return width, step, nsym


# --------------------------------------------------------------------------
# Encoder / decoder
# --------------------------------------------------------------------------
def _map_ctx(bands, y, x, nb):
    left = bands[y, x - 1] if x > 0 else 0
    top = bands[y - 1, x] if y > 0 else 0
    return int(left) * nb + int(top)


def encode(img, thresholds=(64, 192), deltas=(2, 0, 2)):
    assert img.dtype == np.uint8 and img.ndim == 2
    nb = len(thresholds) + 1
    assert len(deltas) == nb
    h, w = img.shape
    ranges = band_edges(thresholds)
    bands = np.digitize(img, thresholds).astype(np.uint8)

    map_models = [Model(nb) for _ in range(nb * nb)]
    off_models = {}
    enc_map, enc_off = Encoder(), Encoder()
    q_arr = np.zeros((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            b = int(bands[y, x])
            enc_map.encode(map_models[_map_ctx(bands, y, x, nb)], b)

    for y in range(h):
        for x in range(w):
            b = int(bands[y, x])
            lo, hi = ranges[b]
            d = deltas[b]
            width, step, nsym = quant_params(lo, hi, d)
            q = (int(img[y, x]) - lo + d) // step
            q_arr[y, x] = q
            if x > 0 and bands[y, x - 1] == b:
                ctx = (b, min(3, int(q_arr[y, x - 1]) * 4 // nsym))
            else:
                ctx = (b, 4)
            m = off_models.setdefault(ctx, Model(nsym))
            enc_off.encode(m, q)

    smap, soff = enc_map.finish(), enc_off.finish()
    header = b"PVS1" + struct.pack(">HHB", h, w, nb)
    header += bytes(thresholds) + bytes(deltas) + struct.pack(">I", len(smap))
    return header + smap + soff


def decode(data):
    assert data[:4] == b"PVS1"
    h, w, nb = struct.unpack(">HHB", data[4:9])
    p = 9
    thresholds = tuple(data[p:p + nb - 1]); p += nb - 1
    deltas = tuple(data[p:p + nb]); p += nb
    (lmap,) = struct.unpack(">I", data[p:p + 4]); p += 4
    dec_map, dec_off = Decoder(data[p:p + lmap]), Decoder(data[p + lmap:])
    ranges = band_edges(thresholds)

    bands = np.zeros((h, w), dtype=np.uint8)
    map_models = [Model(nb) for _ in range(nb * nb)]
    for y in range(h):
        for x in range(w):
            bands[y, x] = dec_map.decode(map_models[_map_ctx(bands, y, x, nb)])

    out = np.zeros((h, w), dtype=np.uint8)
    q_arr = np.zeros((h, w), dtype=np.int32)
    off_models = {}
    for y in range(h):
        for x in range(w):
            b = int(bands[y, x])
            lo, hi = ranges[b]
            d = deltas[b]
            width, step, nsym = quant_params(lo, hi, d)
            if x > 0 and bands[y, x - 1] == b:
                ctx = (b, min(3, int(q_arr[y, x - 1]) * 4 // nsym))
            else:
                ctx = (b, 4)
            m = off_models.setdefault(ctx, Model(nsym))
            q = dec_off.decode(m)
            q_arr[y, x] = q
            out[y, x] = lo + min(q * step, width - 1)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim_global(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def synthetic_image(n=128, seed=0):
    """Surveillance-like scene: smooth dark/bright background + objects + noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = 40 + 120 * (yy / n) + 20 * np.sin(xx / 17.0)
    img[n // 4:n // 2, n // 5:n // 3] = 215            # bright object
    img[int(n * .6):int(n * .85), int(n * .55):int(n * .8)] = 100  # mid object
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def report(img, thresholds, deltas):
    blob = encode(img, thresholds, deltas)
    rec = decode(blob)
    raw = img.size
    err = np.abs(img.astype(int) - rec.astype(int))
    allowed = np.array(deltas)[np.digitize(img, thresholds)]
    print(f"thresholds={thresholds} deltas={deltas}")
    print(f"  size            : {len(blob)} bytes  ({8 * len(blob) / raw:.3f} bpp)")
    print(f"  compression     : {raw / len(blob):.2f}:1")
    print(f"  PSNR            : {psnr(img, rec):.2f} dB")
    print(f"  SSIM (global)   : {ssim_global(img, rec):.4f}")
    print(f"  max |error|     : {err.max()}")
    print(f"  bound respected : {bool(np.all(err <= allowed))}")
    z = len(zlib.compress(img.tobytes(), 9))
    print(f"  zlib baseline   : {z} bytes ({raw / z:.2f}:1)")
    try:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG", optimize=True)
        print(f"  PNG baseline    : {buf.tell()} bytes ({raw / buf.tell():.2f}:1)")
    except Exception:
        pass
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[64, 192])
    ap.add_argument("--deltas", type=int, nargs="+", default=[2, 0, 2])
    ap.add_argument("--otsu", action="store_true", help="auto 3-band thresholds")
    a = ap.parse_args()

    if a.image:
        from PIL import Image
        img = np.array(Image.open(a.image).convert("L"))
    else:
        img = synthetic_image()
    th = multi_otsu_3(img) if a.otsu else tuple(a.thresholds)

    print("--- lossless (all deltas = 0) ---")
    r0 = report(img, th, tuple([0] * (len(th) + 1)))
    assert np.array_equal(r0, img)
    print("--- near-lossless (critical band lossless) ---")
    report(img, th, tuple(a.deltas))
