"""
Procedural "handwriting" augmentation for hieroglyph line drawings.

Takes a clean line drawing (e.g. skeleton_simplify output) and
re-renders it as if a human drew it: skeletonize to a centerline, add elastic
wobble (shaky hand), re-stroke with a spatially-varying pen width, modulate ink
darkness (pen pressure), introduce small gaps (pen lifts) and slight affine
jitter. Each call with a new seed yields a distinct human-like variant.

Usage:
    python handwriting_augment.py <in_line_drawing.png> <out_prefix> [n_variants]
"""
import sys, numpy as np, cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import skeletonize


def _smooth_noise(shape, sigma, rng):
    n = gaussian_filter(rng.random(shape), sigma)
    return (n - n.min()) / (n.max() - n.min() + 1e-8)


def _elastic(field, alpha, sigma, rng):
    h, w = field.shape
    dx = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    warped = map_coordinates(field.astype(np.float32),
                             [(yy + dy).ravel(), (xx + dx).ravel()],
                             order=1, mode="constant").reshape(h, w)
    return warped


def handwrite(line_img_gray, rng,
              base_r=1.5, width_var=1.8, width_sigma=14,
              alpha=7.0, elastic_sigma=5.0,
              pressure_lo=0.6, gap_q=0.02, drop_stroke_p=0.2,
              ink_blur=0.5):
    """line_img_gray: HxW uint8, dark lines on white. returns uint8 dark-on-white."""
    h, w = line_img_gray.shape
    line = (line_img_gray < 128).astype(np.uint8)            # 1 = ink

    # despeckle the source line mask (drop tiny specks that become stipple)
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(line, 8)
    for k in range(1, nc):
        if stats[k, cv2.CC_STAT_AREA] < 10:
            line[lab == k] = 0

    # warp the SOLID mask (stays connected), THEN skeletonize -> continuous centerline
    warped = (_elastic(line.astype(np.float32), alpha, elastic_sigma, rng) > 0.3)
    skel = skeletonize(warped).astype(np.uint8)

    # optional: drop one connected stroke component for variety (never the
    # largest -- a single glyph is one dominant component, so dropping it would
    # erase the whole sign; only drop when there are >=2 strokes to choose from)
    if rng.random() < drop_stroke_p:
        ncc, clab = cv2.connectedComponents(skel)
        sizes = [(clab == c).sum() for c in range(1, ncc)]
        if len(sizes) >= 2:
            biggest = 1 + int(np.argmax(sizes))
            cand = [c for c in range(1, ncc) if c != biggest]
            skel[clab == rng.choice(cand)] = 0

    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return np.full((h, w), 255, np.uint8)

    # contiguous pen-lift gaps: remove skeleton points in a few low-noise blobs
    gapf = _smooth_noise((h, w), 4, rng)
    keep = gapf[ys, xs] > np.quantile(gapf[ys, xs], gap_q)
    ys, xs = ys[keep], xs[keep]

    # spatially-varying pen width (overlapping disks along the centerline)
    radius = base_r + width_var * _smooth_noise((h, w), width_sigma, rng)
    canvas = np.zeros((h, w), np.float32)
    for y, x in zip(ys, xs):
        cv2.circle(canvas, (int(x), int(y)), max(1, int(round(radius[y, x]))), 1.0, -1)

    pressure = pressure_lo + (1 - pressure_lo) * _smooth_noise((h, w), 10, rng)  # ink darkness
    ink = np.clip(canvas * pressure, 0, 1)
    if ink_blur > 0:
        ink = gaussian_filter(ink, ink_blur)
    out = (255 * (1 - ink)).astype(np.uint8)                  # dark-on-white

    # slight affine jitter (shaky framing / proportions)
    ang = rng.uniform(-7, 7); sh = rng.uniform(-0.06, 0.06); sc = rng.uniform(0.93, 1.07)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 1] += sh
    out = cv2.warpAffine(out, M, (w, h), borderValue=255,
                         flags=cv2.INTER_LINEAR)
    return out


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    g = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        out = handwrite(g, rng)
        cv2.imwrite(f"{prefix}_hw{i+1}.png", out)
    print(f"wrote {n} variants for {src}")


if __name__ == "__main__":
    main()
