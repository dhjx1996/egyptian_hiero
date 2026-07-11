"""
Dictionary-app demo: draw a sign in the browser, get top-k canonical matches.

Stdlib HTTP server + the trained matcher; no extra deps. For local/dev use only
(no auth). On JupyterHub, open a terminal port-forward or use jupyter-server-proxy
(https://<hub>/user/<you>/proxy/<port>/).

    ./.venv/bin/python demo_server.py --ckpt runs/default/best.pt \
        --index runs/default/index.npz --port 8787

runs/default is a symlink to the current production run.
"""
import argparse
import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from match import Matcher, D_CSV

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hieroglyph matcher</title><style>
 body{font-family:system-ui,sans-serif;margin:1.2em;background:#faf8f4;color:#222}
 h1{font-size:1.2em} .row{display:flex;gap:1.5em;flex-wrap:wrap}
 canvas{border:2px solid #b09b6b;border-radius:8px;background:#fff;touch-action:none}
 button{font-size:1em;padding:.45em 1.1em;margin:.4em .4em 0 0;border:1px solid #b09b6b;
        border-radius:6px;background:#fff;cursor:pointer} button:hover{background:#f0e8d8}
 #res{min-width:280px;max-width:420px} .hit{display:flex;align-items:center;gap:.8em;
      padding:.35em .5em;border-bottom:1px solid #e6ddc8}
 .hit img{height:52px;background:#fff;border:1px solid #eee;border-radius:4px}
 .hit .code{font-weight:700;min-width:4em} .hit .glyph{font-size:1.6em;min-width:1.4em}
 .hit .score{color:#7a6a45;font-size:.85em} .hit .desc{font-size:.85em;color:#555}
</style></head><body>
<h1>Draw a hieroglyph &rarr; dictionary match</h1>
<div class="row"><div>
 <canvas id="c" width="360" height="360"></canvas><br>
 <button id="clear">Clear</button><button id="undo">Undo</button>
 <button id="go" style="background:#e8dcc0"><b>Match</b></button>
 pen <input id="pen" type="range" min="2" max="14" value="6">
</div><div id="res"><i>draw on the canvas, then Match</i></div></div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');let strokes=[],cur=null;
ctx.lineCap=ctx.lineJoin='round';
function redraw(){ctx.fillStyle='#fff';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.strokeStyle='#111';for(const s of strokes){ctx.lineWidth=s.w;ctx.beginPath();
 s.pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();}}
function pos(e){const r=cv.getBoundingClientRect();
 return[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];}
cv.addEventListener('pointerdown',e=>{cur={w:+document.getElementById('pen').value,pts:[pos(e)]};
 strokes.push(cur);cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(cur){cur.pts.push(pos(e));redraw();}});
addEventListener('pointerup',()=>cur=null);
document.getElementById('clear').onclick=()=>{strokes=[];redraw();};
document.getElementById('undo').onclick=()=>{strokes.pop();redraw();};
document.getElementById('go').onclick=async()=>{
 const res=document.getElementById('res');res.innerHTML='<i>matching...</i>';
 const r=await fetch('match',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({image:cv.toDataURL('image/png')})});
 const hits=await r.json();
 res.innerHTML=hits.length?hits.map(h=>
  `<div class="hit"><img src="glyph/${h.label}.png" onerror="this.style.display='none'">
   <span class="glyph">${h.char||''}</span><span class="code">${h.label}</span>
   <span class="score">${(100*h.score).toFixed(1)}%</span>
   <span class="desc">${h.description||''}</span></div>`).join(''):'no match';};
redraw();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    matcher = None
    canonical = D_CANON
    top = 8

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, PAGE.encode())
        m = re.match(r"^.*/glyph/([^/]+)\.png$", p)
        if m and LABEL_RE.match(m.group(1)):
            f = os.path.join(self.canonical, m.group(1) + ".png")
            if os.path.isfile(f):
                return self._send(200, open(f, "rb").read(), "image/png")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self.path.rstrip("/").endswith("match"):
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            b64 = data["image"].split(",", 1)[1]
            im = Image.open(io.BytesIO(base64.b64decode(b64)))
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im.convert("RGBA"))
            g = np.array(im.convert("L"))
            hits = self.matcher.match(g, top=self.top)
            return self._send(200, json.dumps(hits, ensure_ascii=False).encode(),
                              "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001 - report to the client during dev
            return self._send(500, f"error: {e}".encode(), "text/plain")

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/default/best.pt")
    ap.add_argument("--index", default="runs/default/index.npz")
    ap.add_argument("--canonical", default=D_CANON, help="glyph thumbnails dir")
    ap.add_argument("--csv", default=D_CSV)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    Handler.matcher = Matcher(args.ckpt, args.index, device=args.device, csv_path=args.csv)
    Handler.canonical = args.canonical
    Handler.top = args.top
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[demo] http://localhost:{args.port}/  (JupyterHub: .../user/<you>/proxy/{args.port}/)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
