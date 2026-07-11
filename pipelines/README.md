# Pipelines

Two production pipelines built on the repo's data + model stacks. Each has its
own README with full operation instructions; this is the map.

| Pipeline | Question it answers | Entry point |
|---|---|---|
| **`generation/`** | "Give me handwriting samples of symbol X" | `python3 pipelines/generation/generate.py` |
| **`matching/`** | "Which symbol did the user draw?" (dictionary app) | `pipelines/matching/match.py` / `demo_server.py` |

**Proof it works:** verified numbers and repro commands live in each pipeline's
README (top-level `README.md` has the summary); small eval/stress-test JSONs
and prototype indexes for the production matcher live under
`matching/runs/` (see `matching/README.md` §"File inventory").

Both are **script-agnostic**: they take a *canonical glyph inventory* (a folder
of `<CLASS>.png`) plus a *handwriting corpus* (a folder tree `<CLASS>/*.png`) and
never hard-code Gardiner signs. Point them at Mayan glyphs (or anything else)
and the same commands work — see the "Other scripts / languages" section in each
README.

They also compose: `generation` writes `<out>/<engine>/<CLASS>/*.png`, which is
exactly the labeled-tree layout `matching/evaluate.py --probe-dir` consumes — so
the matcher doubles as a recognizability metric for generated handwriting.

```
canonical glyphs (utf-pngs, one png per symbol)     handwriting corpora
        │                                           (Hand-drawn dataset, app users, ...)
        │                                                   │
        ├──► generation: procedural engine (CPU, now)       │
        │      skeleton → wobble → variants                 │
        ├──► generation: One-DM engine (GPU)  ◄─ fine-tune ─┤
        │      prep_dataset.py → train.py → generate.py     │
        │                                                   │
        └──► matching: train_encoder.py  ◄──────────────────┘
               build_index.py → match.py / demo_server.py / evaluate.py
```

Environments: `generation` reuses the existing `misc/.venv` (procedural) and
`One-DM/.venv` (diffusion); `matching` has its own `pipelines/matching/.venv`
(`bash pipelines/matching/setup_env.sh`, wired into `misc/resetup.sh`).

Heavy training belongs on the HPC A100s — every README has a "Training on the
HPC" section. This JupyterHub box (Tesla T4) is for smoke tests and demos; all
commands below were verified on it end-to-end.
