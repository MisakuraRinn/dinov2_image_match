import os
import json
import numpy as np

from dinov2_numpy import Dinov2Numpy

# ✅ 改这里：导入你自己的 preprocess（你已经写了 resize short side）
# 约定：preprocess(img_path) -> np.ndarray, shape (1, 3, H, W), float32
from preprocess_image import resize_short_side  # <- 把文件名/函数名换成你自己的


def norm_path(p: str) -> str:
    """done.jsonl 里是 Windows 风格 images\\xxx.jpg，这里统一成当前系统可用的路径"""
    return os.path.normpath(p)


def main():
    done_jsonl = "images/done.jsonl"          # 你的下载日志
    weights_path = "vit-dinov2-base.npz"      # DINOv2 权重
    out_dir = "features_dinov2"               # 特征输出目录
    out_index = os.path.join(out_dir, "features_index.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    # init model
    weights = np.load(weights_path)
    vit = Dinov2Numpy(weights)

    ok = skip = fail = 0

    with open(done_jsonl, "r", encoding="utf-8") as fin, open(out_index, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            # 只处理下载成功且有本地路径的
            if rec.get("status") != "ok" or not rec.get("path"):
                rec2 = dict(rec)
                rec2["feat_status"] = "skip_not_ok_or_no_path"
                fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                skip += 1
                continue

            img_path = norm_path(rec["path"])
            if not os.path.exists(img_path):
                rec2 = dict(rec)
                rec2["feat_status"] = "missing_file"
                rec2["feat_error"] = f"file_not_found: {img_path}"
                fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                fail += 1
                continue

            # 用 idx 做特征文件名（稳定、可对齐）
            idx = int(rec.get("idx", 0))
            feat_path = os.path.join(out_dir, f"{idx:06d}.npy")

            # 断点续跑：特征已存在就跳过
            if os.path.exists(feat_path) and os.path.getsize(feat_path) > 0:
                rec2 = dict(rec)
                rec2["feat_status"] = "exists"
                rec2["feat_path"] = feat_path
                fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                skip += 1
                continue

            try:
                x = resize_short_side(img_path)  # ✅ 你的预处理（resize short side），不要 crop
                x = np.asarray(x, dtype=np.float32)

                feat = vit(x)
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)

                # # 如果你的 vit 输出是 token 序列 (1, N, C)，通常取 CLS token 当图片特征
                # if feat.ndim == 3:
                #     feat = feat[:, 0, :]  # (1, C)

                np.save(feat_path, feat)

                rec2 = dict(rec)
                rec2["feat_status"] = "ok"
                rec2["feat_path"] = feat_path
                rec2["feat_shape"] = list(feat.shape)
                fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                ok += 1

            except Exception as e:
                rec2 = dict(rec)
                rec2["feat_status"] = "failed"
                rec2["feat_error"] = repr(e)
                fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                fail += 1

            if (ok + skip + fail) % 50 == 0:
                print(f"[progress] ok={ok} skip={skip} fail={fail}")

    print("Done.")
    print(f"ok={ok}, skip={skip}, fail={fail}")
    print("Feature index:", out_index)


if __name__ == "__main__":
    main()
