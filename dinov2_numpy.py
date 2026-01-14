import numpy as np

from scipy.ndimage import zoom

def gelu(x):
    # 激活函数（？）
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class Embeddings:
    def __init__(self, weights):
        """
        NumPy 实现的 Dinov2 Embeddings 层。

        参数：
        - weights: 权重字典，包含：
            - 'cls_token': 形状为 (1, 1, hidden_size)
            - 'position_embeddings': 形状为 (1, num_patches + 1, hidden_size)
        """
        self.hidden_size = 768 # D
        self.patch_size  = 14  # ps

        self.cls_token           = weights["embeddings.cls_token"] # (1, 1, D) 这个是用来存算完后的token的
        self.position_embeddings = weights["embeddings.position_embeddings"] # (1, N+1, D)
        # print(f"N+1={self.position_embeddings.shape[1]}")
        # =37*37+1
        # print(self.position_embeddings.shape)
        self.patch_embed_w       = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T
        self.patch_embed_b       = weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T
        # print("ckp!")

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        print(f"B={B}")
        patches = []
        print(H//self.patch_size)
        print(W//self.patch_size)
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch = pixel_values[:, :, i:i+self.patch_size, j:j+self.patch_size].reshape(B, -1)
                # 截取图片中的对应patch的像素部分，然后展平成一维的（B,3*14*14）
                patches.append(patch)

        patches = np.stack(patches, axis=1)  # shape: (B, num_patches, patch_dim) 把原来的列表转换成在第一维堆起来的新的张量
        return patches

        # ************* ToDo, resize the self.position_embeddings to match input's varying sizes  *************
        # 自适应图片大小
    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        对 position embedding 做 2D 插值，以适配不同输入分辨率（DINOv2 / ViT 标准做法）

        embeddings: (B, N+1, D)
        height, width: 输入图像的 H, W
        """

        # 原始 position embedding
        pos_embed = self.position_embeddings  # (1, N+1, D)

        # 当前 patch 数
        num_patches = embeddings.shape[1] - 1  # 去掉 CLS
        num_pos = pos_embed.shape[1] - 1

        print(f"num_patches={num_patches}")
        print(f"num_patches={num_patches}")
        #16*16
        # 如果 patch 数没变，直接返回
        if num_patches == num_pos:
            return pos_embed

        # -----------------------------
        # 1. 分离 CLS token 和 patch token
        # -----------------------------
        cls_pos_embed   = pos_embed[:, 0:1, :]   # (1, 1, D)
        patch_pos_embed = pos_embed[:, 1:, :]    # (1, N, D)

        dim = patch_pos_embed.shape[-1]

        # -----------------------------
        # 2. 计算原始和目标 patch 网格大小
        # -----------------------------
        orig_size = int(np.sqrt(num_pos))
        assert orig_size * orig_size == num_pos, "position embedding 不是平方数"

        new_h = height // self.patch_size
        new_w = width  // self.patch_size

        # -----------------------------
        # 3. 把 1D patch pos embed 还原成 2D 网格
        # -----------------------------
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim)

        # -----------------------------
        # 4. 对 2D 网格做双线性插值
        # -----------------------------
        zoom_h = new_h / orig_size
        zoom_w = new_w / orig_size

        patch_pos_embed = zoom(
            patch_pos_embed,
            (1, zoom_h, zoom_w, 1),
            order=3,      # bicubic
            prefilter=True
        )

        # -----------------------------
        # 5. 再展平成 1D token 序列
        # -----------------------------
        patch_pos_embed = patch_pos_embed.reshape(1, new_h * new_w, dim)

        # -----------------------------
        # 6. 拼回 CLS token
        # -----------------------------
        new_pos_embed = np.concatenate((cls_pos_embed, patch_pos_embed), axis=1)

        return new_pos_embed
    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values) # (B, C, H, W) -> (B, h*w, C*ps**2), h=H//ps, w=W//ps 这里相当于用一个三维元组（图片数量，分割后的patch包数量，像素数量）
        
        # (B, h*w, C*ps**2) @ (C*ps**2, D) + (1, D) -> (B, h*w, D)
        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b
        
        cls_token  = np.tile(self.cls_token, (B, 1, 1)) # (1, 1, D) -> (B, 1, D)
        embeddings = np.concatenate([cls_token, embeddings], axis=1) # (B, h*w+1, D)
        # print(f"embeddings.shape={embeddings.shape}")
        # print(f"H={H}")
        # print(f"W={W}")
        pos_embed  = self.interpolate_pos_encoding(embeddings, H, W) # (B, N+1, D) -> (B, h*w+1, D)
        
        embeddings = embeddings + pos_embed
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight
        self.bias   = bias
        self.eps    = eps

    def __call__(self, x, ):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale: 
    def __init__(self, lambda1): 
        self.lambda1 = lambda1

    def __call__(self, x): 
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias   = bias

    def __call__(self, x):
        return x @ self.weight.T + self.bias







class SingleHeadAttention:
    def __init__(self, config, prefix, weights):
        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        q = self.q_proj(x) # (B, h*w+1, D)
        k = self.k_proj(x) # (B, h*w+1, D)
        v = self.v_proj(x) # (B, h*w+1, D)
        att = np.matmul(q, k.transpose(0,2,1)) / np.sqrt(self.hidden_size) # (B, h*w+1, h*w+1)
        att = softmax(att)
        out = np.matmul(att, v) # (B, h*w+1, D)
        return self.out_proj(out)

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj   = Linear(q_w, q_b)
        self.k_proj   = Linear(k_w, k_b)
        self.v_proj   = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        """
        x: (B, N, D)  where N = h*w + 1
        return: (B, N, D)
        """
        B, N, D = x.shape
        H = self.num_heads
        Hd = self.head_dim
        assert D == H * Hd, f"hidden_size({D}) != num_heads({H}) * head_dim({Hd})"

        # 1) Q K V : (B, N, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) reshape -> (B, H, N, Hd)
        #    先 (B, N, H, Hd) 再 transpose
        q = q.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)

        # 3) attention scores: (B, H, N, N)
        #    (B,H,N,Hd) @ (B,H,Hd,N)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(Hd)

        # 4) softmax over last dim (keys dimension)
        attn = softmax(scores, axis=-1)

        # 5) weighted sum: (B, H, N, Hd)
        out = np.matmul(attn, v)

        # 6) merge heads back: (B, N, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

        # 7) final linear: (B, N, D)
        return self.out_proj(out)
    # def __call__(self, x):
    #     # ************* ToDo, multi-head attention *************
    #     # vit多头注意力
    #     raise NotImplementedError

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(f"{prefix}", weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks     = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm       = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        pos_embed = self.embeddings(pixel_values)
        for blk in self.blocks:
            pos_embed = blk(pos_embed)
        pos_embed = self.norm(pos_embed)
        return pos_embed[:, 0]
