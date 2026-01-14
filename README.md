# ? README.md

## 基于 DINOv2 的图像相似度检索系统（Project 5）

本项目实现了一个 **基于 DINOv2 Vision Transformer 的图像相似度检索系统**，
涵盖 **数据下载 → 特征提取 → 特征库构建 → Top-K 相似度检索 → Django Web 展示** 的完整流程。

项目同时提供 **命令行检索** 与 **Web 检索系统**，适用于课程实验与教学验证。

---

## ? 项目结构说明

```text
assignments/
├─ download_images.py              # 批量下载图片
├─ preprocess_image.py             # 图像预处理
├─ dinov2_numpy.py                 # DINOv2 / ViT NumPy 推理实现
├─ extract_feats_from_done_jsonl.py# 特征提取脚本
├─ build_feature_bank_npz.py       # 构建特征库
├─ search_topk_to_jsonl.py         # 命令行 Top-K 检索
├─ feature_bank.npz                # 图像特征库
├─ vit-dinov2-base.npz             # DINOv2 权重
├─ demo_data/                      # 示例图片与特征
├─ images/                         # 下载的原始图片
├─ features_dinov2/                # 单图特征与索引文件
└─ django_app/
   ├─ manage.py
   ├─ retrieval/                   # 检索逻辑与模型
   ├─ templates/                   # 前端页面
   └─ static/                      # CSS / JS / 动效
```

---

## ? 环境依赖

* Python 3.8+
* 主要依赖库：

  ```text
  numpy
  pillow
  scipy
  tqdm
  requests
  django
  ```

安装方式（示例）：

```bash
pip install numpy pillow scipy tqdm requests django
```

> ?? 本项目未提供 `requirements.txt`，请根据需要自行安装依赖。

---

## ? 数据下载

原始图像下载链接存储于 `data.csv` 文件中。

运行以下命令批量下载图片：

```bash
python download_images.py
```

* 下载后的图片将保存在 `images/` 目录
* 脚本支持断点续传与下载日志
* 下载状态会写入对应的 jsonl 文件，供后续处理使用

---

## ? 特征提取与特征库构建

### 1?? 提取图像特征

对已下载完成的图片，使用 DINOv2 提取特征：

```bash
python extract_feats_from_done_jsonl.py
```

* 每张图片生成一个 `*.npy` 特征文件（768 维）
* 同时生成 `features_index.jsonl` 索引文件

---

### 2?? 构建特征库

将分散的特征聚合为统一特征库：

```bash
python build_feature_bank_npz.py
```

生成文件：

```text
feature_bank.npz
```

内容包括：

* 归一化后的特征矩阵（N × 768）
* 图片路径、索引等元信息

---

## ? 命令行 Top-K 检索

可通过命令行方式进行图像相似度检索：

```bash
python search_topk_to_jsonl.py
```

功能：

* 计算查询图像特征
* 与特征库进行余弦相似度匹配
* 输出 Top-K 相似结果（jsonl 格式）

---

## ? Django Web 系统运行

### 1?? 数据库迁移

```bash
cd django_app
python manage.py migrate
```

### 2?? 启动服务器

```bash
python manage.py runserver
```

### 3?? 浏览器访问

```text
http://127.0.0.1:8000/
```

---

## ? Web 系统功能

* ? 用户注册 / 登录 / 退出
* ? 图片上传与在线相似度检索
* ? Top-K 检索结果可视化
* ? 用户检索历史记录管理
* ? 动态前端背景与交互特效

---

## ? 核心实现说明

* **DINOv2 推理**：
  使用 NumPy 实现 Vision Transformer 前向传播，不依赖 PyTorch

* **相似度计算**：
  使用标准余弦相似度：
  [
  \text{sim}(x,y)=\frac{x^\top y}{|x||y|}
  ]

* **Web 检索逻辑**：
  封装于 `retrieval/retriever.py`，统一加载特征库并执行搜索

---

## ?? 注意事项

* 确保 `vit-dinov2-base.npz` 与 `feature_bank.npz` 路径正确
* NumPy 推理速度较慢，适合实验与教学，不适合大规模在线部署
* 若数据量较大，建议限制 Top-K 或使用子集测试

---

## ? 项目说明

* 本项目为课程 **Project 5** 实验作业
* 主要用于展示 **视觉特征提取 + 相似度检索 + Web 系统集成** 的完整流程
