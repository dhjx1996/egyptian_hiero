"""
Capped-simplification v2 for clean canonical glyph line-art (utf-pngs).

Canonical Gardiner renderings are thin OUTLINE drawings, so the strokes a person
copies are the CENTERLINES. Pipeline: binarize -> skeletonize -> trace the
skeleton into single open/closed strokes -> Douglas-Peucker simplify each ->
keep the N longest (stroke budget) -> render clean single strokes.

NO handwriting wobble here (baseline only).
"""
import numpy as np, cv2
from PIL import Image, ImageOps
from scipy.ndimage import label
from skimage.morphology import skeletonize

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def to_ink(path, size=512):
    im = Image.open(path).convert("L")
    im = ImageOps.pad(im, (max(im.size),) * 2, color=255)
    g = np.array(im.resize((size, size), Image.LANCZOS))
    return (g < 128).astype(np.uint8), size


def _trace_component(coords):
    """coords: set of (y,x) for one connected skeleton component -> list of paths."""
    cs = set(coords)
    nb = lambda p: [(p[0]+a, p[1]+b) for a, b in N8 if (p[0]+a, p[1]+b) in cs]
    deg = {p: len(nb(p)) for p in cs}
    nodes = [p for p in cs if deg[p] != 2]          # endpoints + junctions

    def walk(prev, cur):
        path = [prev, cur]
        seen = {prev, cur}
        while deg.get(cur, 0) == 2:
            nxt = [q for q in nb(cur) if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if cur in seen or cur in nodes:          # guard against looping forever
                break
            seen.add(cur)
        return path

    strokes, used = [], set()
    if not nodes:                                    # pure loop (e.g. a ring)
        start = next(iter(cs)); n = nb(start)
        if n:
            strokes.append(walk(start, n[0]))
        return strokes
    for nd in nodes:
        for q in nb(nd):
            if (nd, q) in used:
                continue
            p = walk(nd, q)
            for a, b in zip(p, p[1:]):
                used.add((a, b)); used.add((b, a))
            strokes.append(p)
    return strokes


def prune_spurs(skel, prune_len=14, rounds=4):
    """Remove short side-branches (spurs) that a thick stroke's medial axis grows."""
    sk = skel.copy()
    for _ in range(rounds):
        cs = set(map(tuple, np.argwhere(sk)))
        nb = lambda p: [(p[0]+a, p[1]+b) for a, b in N8 if (p[0]+a, p[1]+b) in cs]
        deg = {p: len(nb(p)) for p in cs}
        remove = set()
        for e in [p for p in cs if deg[p] == 1]:            # each endpoint
            path, prev, cur = [e], None, e
            while True:
                nxt = [q for q in nb(cur) if q != prev]
                if len(nxt) != 1 or deg[cur] >= 3 or len(path) > prune_len:
                    break
                prev, cur = cur, nxt[0]; path.append(cur)
            if len(path) <= prune_len and deg.get(cur, 0) >= 3:
                remove.update(path[:-1])                     # drop spur, keep junction
        if not remove:
            break
        for p in remove:
            sk[p] = 0
    return sk


def strokes_of(ink, size, eps_frac=0.012, min_len_frac=0.03):
    skel = prune_spurs(skeletonize(ink > 0))
    lab, n = label(skel, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        for path in _trace_component(set(map(tuple, np.argwhere(lab == c)))):
            pts = np.array(path)[:, ::-1].astype(np.int32)        # (x,y)
            closed = len(path) > 3 and abs(path[0][0]-path[-1][0]) <= 1 and abs(path[0][1]-path[-1][1]) <= 1
            length = cv2.arcLength(pts.reshape(-1, 1, 2), closed)
            if length < min_len_frac * 4 * size:
                continue
            approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps_frac * length, closed)
            out.append((length, closed, approx))
    out.sort(key=lambda t: -t[0])
    return out


def render(strokes, size, n, thickness=4):
    canvas = np.full((size, size), 255, np.uint8)
    for _, closed, poly in strokes[:n]:
        cv2.polylines(canvas, [poly], closed, 0, thickness, cv2.LINE_AA)
    return canvas


def simplify(path, n=99, size=512, eps_frac=0.012, thickness=4):
    ink, size = to_ink(path, size)
    return render(strokes_of(ink, size, eps_frac), size, n, thickness)
