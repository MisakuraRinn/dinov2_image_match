import json
import numpy as np
from datetime import datetime

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side


def ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    return x.reshape(-1)


def l2_normalize_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def search_topk_to_jsonl(
    query_img_path: str,
    k: int,
    feature_bank_npz: str,
    weights_path: str,
    out_jsonl: str,
    crop_size: int = 224,
):
    # ===== 1. load feature bank =====
    bank = np.load(feature_bank_npz, allow_pickle=True)
    feats = bank["feats"].astype(np.float32)          # (N, 768)
    metas = json.loads(str(bank["metas"]))

    # ===== 2. build query feature =====
    vit = Dinov2Numpy(np.load(weights_path))
    q = vit(resize_short_side(query_img_path))
    q = ensure_1d(q).astype(np.float32)
    q = l2_normalize_1d(q)

    # ===== 3. cosine similarity =====
    sims = feats @ q
    k = min(k, sims.shape[0])

    top_idx = np.argpartition(-sims, kth=k-1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # ===== 4. write jsonl =====
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rank, i in enumerate(top_idx, start=1):
            record = {
                "query_image": query_img_path,
                "rank": rank,
                "similarity": float(sims[i]),
                "idx": metas[i]["idx"],
                "img_path": metas[i]["img_path"],
                "feat_path": metas[i]["feat_path"],
                "caption": metas[i]["caption"],
                "url": metas[i]["url"],
                "timestamp": datetime.now().isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] top-{k} written to {out_jsonl}")


if __name__ == "__main__":
    search_topk_to_jsonl(
        query_img_path="./demo_data/query.jpg",
        k=10,
        feature_bank_npz="feature_bank.npz",
        weights_path="vit-dinov2-base.npz",
        out_jsonl="topk.jsonl",
    )
