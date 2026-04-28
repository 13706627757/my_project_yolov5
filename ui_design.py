from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import os

class GarbageUI(QWidget):

    def __init__(self):
        super().__init__()
        self.counts = {'harmful': 0, 'recyclable': 0, 'kitchen': 0, 'other': 0}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('垃圾分类监控本地开发版')
        self.resize(1600, 1000)

        # --- 左侧：视频显示 ---
        self.video_label = QLabel('画面预览区\n(等待按键触发)')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2C3E50; color: white; border-radius: 15px; font-size: 20px;")

        # --- 右侧：统计面板 ---
        right_panel = QVBoxLayout()
        self.lbl_harmful, self.cnt_harmful = self.create_item("有害垃圾", "#E74C3C")
        self.lbl_recyclable, self.cnt_recyclable = self.create_item("可回收物", "#3498DB")
        self.lbl_kitchen, self.cnt_kitchen = self.create_item("厨余垃圾", "#2ECC71")
        self.lbl_other, self.cnt_other = self.create_item("其他垃圾", "#7F8C8D")

        # --- 右侧：识别结果预览 ---
        self.result_title = QLabel("识别结果预览")
        self.result_title.setFont(QFont('微软雅黑', 14, QFont.Bold))
        self.result_title.setAlignment(Qt.AlignCenter)

        self.result_image_label = QLabel('等待触发识别')
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumHeight(260)
        self.result_image_label.setStyleSheet("background-color: #1F2D3D; color: white; border-radius: 12px; border: 1px solid #34495E;")

        self.result_info_label = QLabel('类别: -\n置信度: -')
        self.result_info_label.setAlignment(Qt.AlignCenter)
        self.result_info_label.setFont(QFont('微软雅黑', 12))
        self.result_info_label.setStyleSheet("background: white; border-radius: 8px; padding: 8px;")

        # 模拟物理按键的按钮
        self.trigger_btn = QPushButton("🚀 模拟物理按键触发 (或按空格)")
        self.trigger_btn.setFixedHeight(60)
        self.trigger_btn.setStyleSheet("background-color: #F39C12; color: white; font-weight: bold; font-size: 18px; border-radius: 10px;")

        right_panel.addWidget(self.lbl_harmful)
        right_panel.addWidget(self.lbl_recyclable)
        right_panel.addWidget(self.lbl_kitchen)
        right_panel.addWidget(self.lbl_other)
        right_panel.addSpacing(10)
        right_panel.addWidget(self.result_title)
        right_panel.addWidget(self.result_image_label)
        right_panel.addWidget(self.result_info_label)
        right_panel.addStretch(1)
        right_panel.addWidget(self.trigger_btn)

        # 总布局
        layout = QHBoxLayout()
        layout.addWidget(self.video_label, 7)
        layout.addLayout(right_panel, 3)
        self.setLayout(layout)

    def create_item(self, name, color):
        box = QWidget()
        box.setStyleSheet(f"background: white; border: 2px solid {color}; border-radius: 8px;")
        l = QHBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setFont(QFont('微软雅黑', 12, QFont.Bold))
        count_lbl = QLabel("0")
        count_lbl.setFont(QFont('Arial', 20, QFont.Bold))
        count_lbl.setStyleSheet(f"color: {color}; border: none;")
        l.addWidget(name_lbl)
        l.addStretch()
        l.addWidget(count_lbl)
        box.setLayout(l)
        return box, count_lbl