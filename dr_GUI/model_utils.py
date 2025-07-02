import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import deserialize
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 全局載入 Stage 1 與 Stage 2 模型
_STAGE1_PATH = "model_CNN_2.h5"
_STAGE2_PATH = "model_CNN_4.h5"
_model_stage1 = load_model(_STAGE1_PATH, compile=False)
_model_stage2 = load_model(_STAGE2_PATH, compile=False)

# EfficientNetB5 預處理常用輸入大小為 456x456
_INPUT_SIZE = (224, 224)

def predict_dr_level(image_path: str) -> int:
    """
    兩階段推論：Stage1 二元分類，Stage2 四分類
    回傳值範圍 1~5：
    1 = No DR
    2~5 = DR 分級
    """
    # 1. 載入並預處理
    img = load_img(image_path, target_size=_INPUT_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 2. Stage1 二元分類推論
    p1 = _model_stage1.predict(x)
    print(f"[DEBUG] Stage 1 prediction (DR / No DR): {p1}")  # ✅ 印出 Stage1 機率

    if p1.shape[-1] == 1:
        label1 = int(np.round(p1[0][0]))
    else:
        label1 = int(np.argmax(p1, axis=1)[0])

    print(f"[DEBUG] Stage 1 result: {label1}")  # ✅ 印出 Stage1 分類結果

    if label1 == 1:
        return 0  # No DR

    # 3. Stage2 四分類推論
    p2 = _model_stage2.predict(x)
    print(f"[DEBUG] Stage 2 prediction (DR severity): {p2}")  # ✅ 印出 Stage2 機率

    label2 = int(np.argmax(p2, axis=1)[0])  # 0~3
    print(f"[DEBUG] Stage 2 result (mapped): {label2 + 1}")  # ✅ 印出 Stage2 結果轉換後

    return label2 + 1  # map 0→2, 1→3, 2→4, 3→5
