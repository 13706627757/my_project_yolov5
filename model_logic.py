import torch
import cv2


class GarbageDetector:

    def __init__(self, model_path,select_device):  #加载垃圾模型文件
        print("🧠 正在加载 YOLO 权重...")
        # 本地 PC 建议也先用 CPU 测试，速度快
        self.model = torch.hub.load('./', 'custom', path=model_path, source='local', device=select_device)
        self.model.conf = 0.0 # 设置一个置信度门槛


    def predict(self, frame):
        """输入 OpenCV 图片，返回 (画好的图, 类别名称, 置信度)"""
        results = self.model(frame)
        
        # 获取结果表格
        df = results.pandas().xyxy[0]
        label = "none"
        confidence = 0.0
        if not df.empty:
            label = df.iloc[0]['name'] # 获取第一个目标的名称
            confidence = float(df.iloc[0]['confidence'])

        # 获取画好框的图片
        rendered_img = results.render()[0]
        return rendered_img, label, confidence