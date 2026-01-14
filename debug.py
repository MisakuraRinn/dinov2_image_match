import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop,resize_short_side


def l2norm(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def cosine(a, b, axis=-1, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(a * b, axis=axis) / ((np.linalg.norm(a, axis=axis) + eps) * (np.linalg.norm(b, axis=axis) + eps))

def summarize_diff(name, pred, ref):
    pred = np.asarray(pred)
    ref  = np.asarray(ref)

    if pred.shape != ref.shape:
        print(f"\n[{name}] shape mismatch: pred {pred.shape} vs ref {ref.shape}")
        return

    diff = pred - ref
    abs_diff = np.abs(diff)

    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    rmse = np.sqrt((diff * diff).mean())

    # 相对误差（避免 ref=0）
    denom = np.maximum(np.abs(ref), 1e-8)
    rel = abs_diff / denom
    max_rel = rel.max()
    mean_rel = rel.mean()

    # allclose 检查：按你需要调 tol
    ok = np.allclose(pred, ref, rtol=1e-4, atol=1e-5)

    # 余弦相似度（向量）
    if pred.ndim == 1:
        cos = float(cosine(pred, ref))
    else:
        # 如果是 (B,D) 就对每个样本算
        cos = cosine(pred, ref, axis=-1)
        cos = float(np.mean(cos))

    print(f"\n[{name}] allclose={ok}  max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}  rmse={rmse:.6e}")
    print(f"[{name}] max_rel={max_rel:.6e}  mean_rel={mean_rel:.6e}  mean_cosine={cos:.8f}")

    # 额外：误差最大的坐标
    flat_idx = int(abs_diff.reshape(-1).argmax())
    pred_flat = pred.reshape(-1)
    ref_flat  = ref.reshape(-1)
    print(f"[{name}] worst_idx={flat_idx} pred={pred_flat[flat_idx]:.6e} ref={ref_flat[flat_idx]:.6e} abs_diff={abs_diff.reshape(-1)[flat_idx]:.6e}")

def load_reference(path):
    ref = np.load(path, allow_pickle=True)
    # 处理 np.save 保存 dict/list 的情况
    if isinstance(ref, np.ndarray) and ref.dtype == object:
        obj = ref.item()
        return obj
    return ref


# ----------------- 你的推理部分 -----------------
weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

cat_pixel_values = center_crop("./demo_data/cat.jpg")
# cat_pixel_values = resize_short_side("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values)          # 期望 (1,768) 或 (768,)
cat_feat = np.asarray(cat_feat).reshape(-1)  # 拉平成 (768,)

dog_pixel_values = center_crop("./demo_data/dog.jpg")
# dog_pixel_values=resize_short_side("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values)
dog_feat = np.asarray(dog_feat).reshape(-1)
pred = np.stack([cat_feat, dog_feat], axis=0)   # (2,768)

# ----------------- 读取样例输出 -----------------
ref = load_reference("./demo_data/cat_dog_feature.npy")

# 尝试对齐 ref 的结构
if isinstance(ref, dict):
    # 常见：{"cat":..., "dog":...} 或类似 key
    # 你可以按实际 key 改一下
    keys = list(ref.keys())
    print("reference is dict, keys =", keys)
    # 尽量自动猜
    if "cat" in ref and "dog" in ref:
        ref_cat = np.asarray(ref["cat"]).reshape(-1)
        ref_dog = np.asarray(ref["dog"]).reshape(-1)
        ref_pair = np.stack([ref_cat, ref_dog], axis=0)
        summarize_diff("cat_pair", pred, ref_pair)
        summarize_diff("cat", cat_feat, ref_cat)
        summarize_diff("dog", dog_feat, ref_dog)
    else:
        # 退化：随便取前两个
        vals = [np.asarray(ref[k]).reshape(-1) for k in keys[:2]]
        ref_pair = np.stack(vals, axis=0)
        summarize_diff("pair", pred, ref_pair)

else:
    ref = np.asarray(ref)
    print("reference array shape =", ref.shape, "dtype =", ref.dtype)

    # 常见三种： (2,768) / (768,2) / (1536,)
    if ref.shape == (2, 768):
        ref_pair = ref
    elif ref.shape == (768, 2):
        ref_pair = ref.T
    elif ref.size == 2 * 768:
        ref_pair = ref.reshape(2, 768)
    else:
        ref_pair = None

    if ref_pair is None:
        print("Unrecognized reference shape; printing first few numbers to inspect:")
        print(ref.reshape(-1)[:10])
    else:
        summarize_diff("pair", pred, ref_pair)
        summarize_diff("cat", cat_feat, ref_pair[0])
        summarize_diff("dog", dog_feat, ref_pair[1])

# （可选）把你自己的输出也保存下来，方便反复对比
np.save("./demo_data/my_cat_dog_feature.npy", pred)
print("\nSaved my output to demo_data/my_cat_dog_feature.npy")
