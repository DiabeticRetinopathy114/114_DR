import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from model_utils import predict_dr_level

# -----------------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None

    if request.method == "POST":
        # 1) 檢查檔案是否存在
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        # 2) 存檔到 uploads/
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # 3) 呼叫兩階段推論
        level = predict_dr_level(save_path)  # 回傳 1~5
        desc_map = {
            1: "No DR (No Diabetic Retinopathy)",
            2: "Mild DR",
            3: "Moderate DR (Moderate Diabetic Retinopathy)",
            4: "Severe DR",
            5: "Proliferative DR"
        }
        result = {"level": level, "desc": desc_map[level]}

    return render_template("index.html", result=result, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)