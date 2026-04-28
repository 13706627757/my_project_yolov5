import threading

from PyQt5.QtCore import QThread, pyqtSignal

try:
    import serial
except ImportError:
    serial = None


class SerialTriggerListener(QThread):
    """Listen serial bytes and emit a signal when trigger char is received."""

    TRIGGER_CHAR = 't'

    trigger_received = pyqtSignal()
    message_received = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, port='/dev/ttyTHS1', baudrate=115200, timeout=0.2, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._running = True
        self._serial = None
        self._write_lock = threading.Lock()

    def run(self):
        if serial is None:
            self.status.emit('未安装 pyserial，请先执行: pip install pyserial')
            return

        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.status.emit(f'串口已连接: {self.port} @ {self.baudrate}')

            while self._running:
                data = self._serial.read(1)
                if not data:
                    continue

                ch = data.decode('utf-8', errors='ignore').lower()
                if ch:
                    self.message_received.emit(ch)
                if ch == self.TRIGGER_CHAR:
                    self.status.emit('串口收到触发字符 t，准备执行推理')
                    self.trigger_received.emit()
        except Exception as e:
            self.status.emit(f'串口监听异常: {e}')
        finally:
            if self._serial is not None and self._serial.is_open:
                self._serial.close()
                self.status.emit('串口已关闭')

    def stop(self):
        self._running = False
        self.wait(1500)

    def send_text(self, text):
        """Send text to the serial peer from any thread."""
        if self._serial is None or not self._serial.is_open:
            self.status.emit('串口未打开，发送失败')
            return False

        data = str(text).encode('utf-8')
        with self._write_lock:
            try:
                self._serial.write(data)
                self._serial.flush()
                self.status.emit(f'已发送串口数据: {text}')
                return True
            except Exception as e:
                self.status.emit(f'串口发送失败: {e}')
                return False
