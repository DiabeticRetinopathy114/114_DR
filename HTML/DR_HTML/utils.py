import numpy as np
from PIL import Image
import cv2

# 1) 讀取並 resize
def load_and_resize(image_bytes, target_size=(224,224)):
    img = Image.open(image_bytes).convert('RGB')
    img = img.resize(target_size)
    return np.array(img)

# 2) 正規化 & 加 batch 維度
def preprocess(img_array):
    # 將像素值 [0,255] → [0,1]
    img = img_array.astype('float32') / 255.0
    # 如果模型要做其他標準化（如 ImageNet 預處理），在這裡加
    # from tensorflow.keras.applications.resnet50 import preprocess_input
    # img = preprocess_input(img)
    return np.expand_dims(img, axis=0)  # (1, H, W, 3)

# 3) 將模型 raw output（如 softmax 機率）轉成類別
def postprocess(preds):
    # 假設 preds.shape == (1,5)，五個等級的機率
    class_idx = int(np.argmax(preds, axis=1)[0])
    labels = {
        0: "無 DR (No DR)",
        1: "輕度 NPDR (Mild NPDR)",
        2: "中度 NPDR (Moderate NPDR)",
        3: "重度 NPDR (Severe NPDR)",
        4: "增殖性 DR (PDR)"
    }
    return class_idx, labels[class_idx]