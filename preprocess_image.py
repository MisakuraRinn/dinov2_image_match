import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    print("ckp1")
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

# ************* ToDo, resize short side *************
# def resize_short_side(img_path, target_size=224):
#     print("ckp2")
#     # Step 1: load image
#     image = Image.open(img_path).convert("RGB")

#     # Step 2: resize so that the shorter side == target_size
#     # and more, ensure both sides are multiples of patch size, e.g., 14
#     # 应该是缩放，将短边的大小缩放到224，然后确保长边的大小也是14的倍数，对14取模后余下的部分去掉
    
    
#     # Step 3: to_numpy
#     image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

#     # Step 4: norm
#     mean = np.array([0.485, 0.456, 0.406])
#     std  = np.array([0.229, 0.224, 0.225])
#     image = (image - mean) / std  # (H, W, C)
#     image = image.transpose(2, 0, 1) # (C, H, W)
#     return image[None] # (1, C, H, W)

def resize_short_side(img_path, target_size=224, patch_size=14):
    # print("ckp2")

    # Step 1: load image
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # Step 2: resize so that the shorter side == target_size
    if w < h:
        scale = target_size / w
        new_w = target_size
        new_h = int(round(h * scale))
    else:
        scale = target_size / h
        new_h = target_size
        new_w = int(round(w * scale))

    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Step 3: ensure long side is multiple of patch_size (crop tail)
    new_w, new_h = image.size

    if new_w > new_h:
        # width is long side
        valid_w = (new_w // patch_size) * patch_size
        image = image.crop((0, 0, valid_w, new_h))
    else:
        # height is long side
        valid_h = (new_h // patch_size) * patch_size
        image = image.crop((0, 0, new_w, valid_h))

    # Step 4: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 5: normalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)

    # Step 6: CHW + batch
    image = image.transpose(2, 0, 1)  # (C, H, W)
    return image[None]  # (1, C, H, W)
