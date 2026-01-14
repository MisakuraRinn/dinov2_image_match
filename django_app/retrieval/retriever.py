import json
import numpy as np

from django.conf import settings

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    return x.reshape(-1)


def _l2_normalize_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


class Retriever:
    def __init__(self, feature_bank_path: str, weights_path: str):
        bank = np.load(feature_bank_path, allow_pickle=True)
        self.feats = bank["feats"].astype(np.float32)
        self.metas = json.loads(str(bank["metas"]))
        self.vit = Dinov2Numpy(np.load(weights_path))

    def search(self, query_img_path: str, k: int):
        q = self.vit(resize_short_side(query_img_path))
        q = _ensure_1d(q).astype(np.float32)
        q = _l2_normalize_1d(q)

        sims = self.feats @ q
        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            meta = self.metas[i]
            results.append({
                "rank": rank,
                "similarity": float(sims[i]),
                "img_path": meta.get("img_path", ""),
                "caption": meta.get("caption", ""),
                "url": meta.get("url", ""),
            })
        return results


_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER

    feature_bank_path = settings.DATA_ROOT / "feature_bank.npz"
    weights_path = settings.DATA_ROOT / "vit-dinov2-base.npz"

    if not feature_bank_path.exists() or not weights_path.exists():
        return None

    _RETRIEVER = Retriever(str(feature_bank_path), str(weights_path))
    return _RETRIEVER
