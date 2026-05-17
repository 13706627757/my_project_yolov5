import sys
import argparse
import warnings
import cv2
import time
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
# 导入我们刚才写的两个模块
from ui_design import GarbageUI

from model_logic import GarbageDetector
from serial_listener import SerialTriggerListener

# 抑制 CUDA 相关警告（Jetson 上用 CPU 加载模型时会出现）
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.meshgrid.*')


# =========================
# 通用逻辑（本地 + Jetson）
# =========================
def label_to_serial_code(label):
    mapping = {
        'harmful': '0',
        'recyclable': '1',
        'kitchen': '2',
        'other': '3',
    }
    return mapping.get(label)


def infer_once(detector, cap):
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    img, label, confidence = detector.predict(frame)
    result_code = label_to_serial_code(label)
    return img, label, confidence, result_code


def infer_with_retry(detector, cap, max_attempts=3, retry_delay_sec=0.1):
    """连续推理多次，直到识别出类别；否则返回 fallback code 4。"""
    last_img = None
    last_label = None
    last_confidence = 0.0

    for attempt in range(1, max_attempts + 1):
        img, label, confidence, result_code = infer_once(detector, cap)
        if img is None:
            print(f'❌ 第 {attempt} 次抓图失败')
            return None, None, 0.0, None

        last_img = img
        last_label = label
        last_confidence = confidence

        if result_code is not None:
            return img, label, confidence, result_code

        if attempt < max_attempts:
            print(f'⚠️ 第 {attempt} 次未识别到结果，{int(retry_delay_sec * 1000)}ms 后重试...')
            
            time.sleep(retry_delay_sec)

    return last_img, last_label, last_confidence, '4'


class StreamDisplayThread:
    """后台持续读取摄像头并回调给主线程更新显示"""
    def __init__(self, cap, callback, interval_ms=33):
        self.cap = cap
        self.callback = callback
        self.interval_ms = interval_ms
        self._running = True
        from threading import Thread
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        import time
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                self.callback(frame)
            time.sleep(self.interval_ms / 1000.0)
    
    def stop(self):
        self._running = False

class MainController(GarbageUI):
    def __init__(self, camera_index=0, model_path='my_weight/best-100.pt', device='cpu', serial_port=None, baudrate=115200):
        super().__init__()
        self.warning_dialog = None
        self.serial_listener = None
        # 1. 实例化大脑
        self.detector = GarbageDetector(model_path, device)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print("❌ 摄像头打开失败，请检查设备连接。")

        # 2. 启动实时摄像头显示线程
        self.stream_thread = StreamDisplayThread(self.cap, self.update_stream_frame)

        if serial_port:
            self.serial_listener = SerialTriggerListener(port=serial_port, baudrate=baudrate)
            self.serial_listener.trigger_received.connect(self.run_detection)
            self.serial_listener.message_received.connect(self.on_serial_message)
            self.serial_listener.status.connect(print)
            self.serial_listener.start()

        # 3. 绑定按钮点击事件
        self.trigger_btn.clicked.connect(self.run_detection)
        
    # 处理键盘事件 (按空格也能触发)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.run_detection()
        elif event.key() == Qt.Key_K:
            self.show_recyclable_warning()
        elif event.key() == Qt.Key_O:
            self.show_full_warning('预警', '垃圾桶已满载，请及时清理或更换垃圾袋。', '#E74C3C')

    def show_recyclable_warning(self):
        self.show_full_warning('预警', '可回收垃圾桶已满载，请及时清理或更换垃圾袋。', '#3498DB')

    def show_full_warning(self, title_text, message_text, accent_color):
        if self.warning_dialog is not None and self.warning_dialog.isVisible():
            self.warning_dialog.raise_()
            self.warning_dialog.activateWindow()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('预警')
        dialog.setModal(True)
        dialog.setFixedSize(560, 320)
        dialog.setStyleSheet(
            'QDialog { background: #1E1E1E; }'
            'QLabel { color: white; }'
            'QPushButton { background: #E74C3C; color: white; border: none; border-radius: 10px; padding: 10px 18px; font-size: 16px; }'
            'QPushButton:hover { background: #FF6B5B; }'
        )

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f'font-size: 30px; font-weight: 700; color: {accent_color};')

        message = QLabel(message_text)
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        message.setStyleSheet('font-size: 18px; color: #F2F2F2;')

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet('color: #444; background: #444; max-height: 1px;')

        close_btn = QPushButton('确认')
        close_btn.setFixedHeight(48)
        close_btn.clicked.connect(dialog.accept)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(18)
        layout.addWidget(title)
        layout.addWidget(separator)
        layout.addWidget(message)
        layout.addStretch(1)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)

        dialog.finished.connect(lambda _: setattr(self, 'warning_dialog', None))
        self.warning_dialog = dialog
        dialog.show()

    def on_serial_message(self, message):
        if not message or message in ('\r', '\n'):
            return

        print(f'串口 RX <- {message}')
        if message.upper() == 'K':
            self.show_recyclable_warning()

    def update_stream_frame(self, frame):
        """实时更新摄像头画面到界面"""
        self.show_image(frame)

    def update_detection_result(self, img, label, confidence):
        """把识别后的结果图显示到右侧，并展示类别和置信度"""
        self.show_result_image(img)
        self.result_info_label.setText(f'类别: {label}\n置信度: {confidence:.2%}')

    def run_detection(self):
        """这就是被按键触发的核心功能"""
        print("📡 收到触发指令，正在抓图...")
        img, label, confidence, result_code = infer_with_retry(self.detector, self.cap)
        if img is not None:
            # 调用大脑进行检测
            # 更新右侧识别结果
            self.update_detection_result(img, label, confidence)
            
            # 更新数量逻辑
            if label in self.counts:
                self.counts[label] += 1
                self.update_counter(label)

            if result_code is not None:
                print(f"推理结果: label={label}, result_code={result_code}")
                if self.serial_listener and self.serial_listener.isRunning():
                    ok = self.serial_listener.send_text(result_code)
                    if ok:
                        print(f"串口 TX -> {result_code}")
                    else:
                        print(f"⚠️ 串口发送失败，result_code={result_code}")
            else:
                print(f"⚠️ 未识别到有效类别，label={label}")
        else:
            print("❌ 抓图失败，未执行推理。")

    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def show_result_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.result_image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.result_image_label.width(), self.result_image_label.height(), Qt.KeepAspectRatio))

    def update_counter(self, label):
        val = str(self.counts[label])
        if label == 'harmful': self.cnt_harmful.setText(val)
        elif label == 'recyclable': self.cnt_recyclable.setText(val)
        elif label == 'kitchen': self.cnt_kitchen.setText(val)
        elif label == 'other': self.cnt_other.setText(val)

    def closeEvent(self, event):
        if hasattr(self, 'stream_thread'):
            self.stream_thread.stop()
        if hasattr(self, 'serial_listener') and self.serial_listener and self.serial_listener.isRunning():
            self.serial_listener.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


# =========================
# 本地模式（笔记本终端）
# 实时显示摄像头，按 t 键触发推理
# 输出 0-3 到终端，q 键退出
# =========================
def run_gui_mode(mode_name, qt_args):
    print(f'================ {mode_name} 模式 GUI ================')
    print('说明: Qt 界面实时显示摄像头画面，按按钮或空格键触发推理')
    app = QApplication([sys.argv[0]] + qt_args)
    if mode_name == 'jetson':
        ctrl = MainController(serial_port='/dev/ttyTHS1', baudrate=115200)
    else:
        ctrl = MainController() 
    ctrl.show()
    sys.exit(app.exec_())  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Garbage detection UI')
    parser.add_argument('--mode', choices=['jetson', 'local'], default='local', help='两种模式使用同一套 Qt 界面与识别流程，仅保留命名区分')
    args, qt_args = parser.parse_known_args()
    run_gui_mode(args.mode, qt_args)