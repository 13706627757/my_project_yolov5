import sys
import argparse
import warnings
import cv2
import time
from PyQt5.QtWidgets import QApplication
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
    def __init__(self, serial_port='/dev/ttyTHS1', baudrate=115200):
        super().__init__()
        # 1. 实例化大脑
        self.detector = GarbageDetector('best.pt', 'cpu')
        self.cap = cv2.VideoCapture(0) # 打开本地摄像头

        if not self.cap.isOpened():
            print("❌ 摄像头打开失败，请检查设备连接。")

        # 2. 启动实时摄像头显示线程
        self.stream_thread = StreamDisplayThread(self.cap, self.update_stream_frame)

        self.serial_listener = None
        if serial_port:
            # 3. 启动串口监听：默认 Jetson Nano 可用 /dev/ttyTHS1，本地可改成 COMx
            print(f"串口启动参数: port={serial_port}, baudrate={baudrate}")
            self.serial_listener = SerialTriggerListener(port=serial_port, baudrate=baudrate)
            self.serial_listener.trigger_received.connect(self.run_detection)
            self.serial_listener.message_received.connect(self.on_serial_message)
            self.serial_listener.status.connect(print)
            self.serial_listener.start()
        
        # 4. 绑定按钮点击事件
        self.trigger_btn.clicked.connect(self.run_detection)
        
    # 处理键盘事件 (按空格也能触发)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.run_detection()

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

            # 检测结束后按垃圾类别回发 0-3 给 STM32
            if result_code is not None:
                print(f"推理结果: label={label}, 发送串口结果码={result_code}")
                self.send_serial(result_code)
            else:
                print(f"⚠️ 未识别到有效类别，label={label}，不发送结果码。")
        else:
            print("❌ 抓图失败，未执行推理。")

    def on_serial_message(self, message):
        if message and message != '\r' and message != '\n':
            print(f"串口 RX <- {message}")

    def send_serial(self, text):
        if hasattr(self, 'serial_listener'):
            self.serial_listener.send_text(text)
        else:
            print('串口模块未初始化，无法发送')

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
def run_local_test_mode():
    print('================ 本地模式 local ================')
    print('说明: 实时显示摄像头，按 t 键触发推理，按 q 键退出')
    print('提示: 点击摄像头窗口后再按键')

    detector = GarbageDetector('best.pt', 'cpu')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('❌ 摄像头打开失败，请检查设备连接。')
        return

    try:
        cv2.namedWindow('Local Test - Real-time Stream', cv2.WINDOW_AUTOSIZE)
        print('按 t 键推理，按 q 键退出')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print('❌ 抓图失败')
                break

            cv2.imshow('Local Test - Real-time Stream', frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print('用户退出')
                break
            elif key == ord('t'):
                print('📡 收到触发指令，正在推理...')
                img, label, confidence, result_code = infer_with_retry(detector, cap)
                if img is None:
                    print('❌ 推理失败')
                    continue

                print(f'推理结果: label={label}, confidence={confidence:.2%}')
                if result_code is not None:
                    print(f'发送结果码: {result_code}')
                else:
                    print('⚠️ 未识别到有效类别，不发送结果码。')
    finally:
        cap.release()
        cv2.destroyAllWindows()


# =========================
# Jetson 模式（GUI + 串口）
# 收到串口 t 触发推理
# 推理后发送 0-3 给下位机
# =========================
def run_jetson_mode(serial_port, baudrate, qt_args):
    print('================ Jetson 模式 jetson ================')
    print(f'说明: 串口监听中，port={serial_port}, baudrate={baudrate}')
    app = QApplication([sys.argv[0]] + qt_args)
    ctrl = MainController(serial_port=serial_port, baudrate=baudrate)
    ctrl.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Garbage detection UI with serial trigger')
    parser.add_argument('--mode', choices=['jetson', 'local'], default='jetson', help='local: 笔记本终端输入 t 测试；jetson: 串口 t 触发 + GUI 显示')
    parser.add_argument('--port', default='/dev/ttyTHS1', help='Serial port, e.g. /dev/ttyTHS1 or COM3')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate')
    args, qt_args = parser.parse_known_args()

    if args.mode == 'local':
        run_local_test_mode()
        sys.exit(0)

    run_jetson_mode(serial_port=args.port, baudrate=args.baudrate, qt_args=qt_args)