import sys
import os
import numpy as np
from model_utils import predict_dr_level

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# 從 model_utils 導入預測函式
from model_utils import predict_dr_level

class DRDetectorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetic Retinopathy Detector")
        self.setFixedSize(700, 800)
        self.init_ui()

    def init_ui(self):
        # ------ 文件輸入部分 ------
        lbl_file = QLabel("File Name:")
        self.edit_path = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_image)

        hbox_input = QHBoxLayout()
        hbox_input.addWidget(lbl_file)
        hbox_input.addWidget(self.edit_path)
        hbox_input.addWidget(btn_browse)

        # ------ 圖片顯示區域 ------
        self.lbl_image = QLabel()
        self.lbl_image.setFixedSize(640, 400)
        self.lbl_image.setStyleSheet("border: 1px solid gray; background-color: #EEE;")
        self.lbl_image.setAlignment(Qt.AlignCenter)

        # ------ Detect 按鈕 ------
        btn_detect = QPushButton("Detect")
        btn_detect.setFixedHeight(40)
        btn_detect.clicked.connect(self.run_detection)

        # ------ 結果顯示部分 (1-5) ------
        self.level_labels = []
        hbox_levels = QHBoxLayout()
        for i in range(5):
            lbl = QLabel(str(i+1))  # 顯示 1-5
            lbl.setFixedSize(60, 40)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont("Arial", 12))
            lbl.setStyleSheet("border: 1px solid black; background-color: none;")
            hbox_levels.addWidget(lbl)
            self.level_labels.append(lbl)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setAlignment(Qt.AlignCenter)
        self.lbl_desc.setFont(QFont("Arial", 12))

        # ------ 總佈局 ------
        vbox = QVBoxLayout()
        vbox.addLayout(hbox_input)
        vbox.addSpacing(10)
        vbox.addWidget(self.lbl_image)
        vbox.addSpacing(10)
        vbox.addWidget(btn_detect)
        vbox.addSpacing(20)
        vbox.addLayout(hbox_levels)
        vbox.addWidget(self.lbl_desc)
        vbox.addStretch()
        self.setLayout(vbox)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", os.getcwd(),
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if path:
            self.edit_path.setText(path)
            self.show_image(path)

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_image.setPixmap(pixmap)

    def run_detection(self):
        path = self.edit_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "Please select a valid image file.")
            return

        try:
            level = predict_dr_level(path)  # 0-4
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")
            return

        # 描述映射
        desc_map = {
            0: "No DR (No Diabetic Retinopathy)",
            1: "Mild DR",
            2: "Moderate DR (Moderate Diabetic Retinopathy)",
            3: "Severe DR",
            4: "Proliferative DR"
        }
        # 重置樣式
        for lbl in self.level_labels:
            lbl.setStyleSheet("border: 1px solid black; background-color: none;")
        # 標記結果，level 0 對應第一個 label (顯示 1)
        print("Raw prediction:", path)
        self.level_labels[level].setStyleSheet(
            "border: 2px solid black; background-color: orange;"
        )
        self.lbl_desc.setText(desc_map.get(level, ""))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DRDetectorWindow()
    window.show()
    sys.exit(app.exec_())
