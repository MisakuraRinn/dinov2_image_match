import os, json
import numpy as np
from tqdm import tqdm


def ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    if x.ndim != 1:
        return x.reshape(-1)
    return x


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def build_feature_bank_npz(
    features_index_jsonl: str,
    out_npz: str,
    root_dir: str = ".",
):
    feats = []
    metas = []

    with open(features_index_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Packing features"):
        obj = json.loads(line)

        if obj.get("feat_status") != "ok":
            continue

        feat_path = obj.get("feat_path")
        if not feat_path:
            continue

        feat_path = feat_path.replace("\\", os.sep)
        feat_path = os.path.join(root_dir, feat_path)
        if not os.path.exists(feat_path):
            continue

        feat = np.load(feat_path)
        feat = ensure_1d(feat).astype(np.float32)

        feats.append(feat)
        metas.append({
            "idx": obj.get("idx"),
            "img_path": obj.get("path"),
            "feat_path": obj.get("feat_path"),
            "url": obj.get("url", ""),
            "caption": obj.get("caption", ""),
        })

    feats = np.stack(feats, axis=0)   # (N, 768)
    feats = l2_normalize_rows(feats)

    np.savez_compressed(
        out_npz,
        feats=feats,
        metas=json.dumps(metas, ensure_ascii=False)
    )

    print(f"[OK] feature_bank saved: {out_npz}")
    print(f"     feats shape = {feats.shape}")


if __name__ == "__main__":
    build_feature_bank_npz(
        features_index_jsonl="features_dinov2/features_index.jsonl",
        out_npz="feature_bank.npz",
        root_dir=".",
    )
