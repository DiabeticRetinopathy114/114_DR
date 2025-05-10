from flask import Flask, request, jsonify
from flask_cors import CORS              # 允許跨域請求
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import io

from utils import load_and_resize, preprocess, postprocess

app = Flask(__name__)
CORS(app)  # 允許所有網域呼叫此 API (前端在不同 host 時用得到)

# 1) 載入你的 Keras 模型
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

@app.route('/predict', methods=['POST'])
def predict():
    # 2) 接收前端上傳的檔案
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # 3) 預處理
    img_array = load_and_resize(file.stream, target_size=(224,224))
    model_input = preprocess(img_array)

    # 4) 推論
    preds = model.predict(model_input)  # shape (1,5)

    # 5) 後處理
    class_idx, label = postprocess(preds)

    # 6) 回傳 JSON
    return jsonify({
        'code': class_idx,
        'label': label,
        'confidence': float(np.max(preds))  # 可選：回傳最高機率
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)