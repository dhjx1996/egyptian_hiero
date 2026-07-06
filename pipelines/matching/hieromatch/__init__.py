"""hieromatch: handwritten-symbol -> canonical-glyph matching (Pipeline 2).

Encoder embeds a drawn symbol; recognition = cosine nearest-prototype against an
index of canonical glyph embeddings (open-set: add symbols/scripts by re-running
build_index.py, no retraining needed). See pipelines/matching/README.md.
"""
from .model import HieroEncoder, CosineHead, load_encoder, save_encoder  # noqa: F401
from .data import (load_gray, crop_ink, letterbox, preprocess_array,      # noqa: F401
                   preprocess_path, TrainDataset, EvalDataset, build_items)
