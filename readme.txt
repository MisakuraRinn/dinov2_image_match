1. How to debug
- Complete dinov2_numpy.py
- Run debug.py
- Check output, i.e., compare your extracted features with the reference (./demo_data/cat_dog_feature.npy). Make sure the difference is within a small numerical tolerance.

2. Image retrieval
- Cownload 10,000+ web images (data.csv) to build the gallery set
- Finish 'resize_short_side' in preprocess_image.py. The function must correctly resize images of different resolutions such that the shorter side becomes the target size (e.g., 224). Meanwhile, both sides should be the multiple of 14
- Extract features for all gallery images via your ViT (dinov2_numpy.py)
- When user upload an image, preprocess → extract features, compute similarity with all gallery features (e.g., cosine similarity or L2 distance), and return the Top-10 most similar images as search results