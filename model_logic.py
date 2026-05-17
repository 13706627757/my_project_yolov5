import torch
import cv2
import os


class GarbageDetector:

    def __init__(self, model_path,select_device):  #加载垃圾模型文件
        print("🧠 正在加载 YOLO 权重...")
        # 本地 PC 建议也先用 CPU 测试，速度快
        # 规范化路径并检查空值，避免传入空字符串导致 hubconf 解析成 ' .pt'
        model_path = str(model_path).strip()
        if model_path == '':
            raise ValueError('模型权重路径为空，请通过 --weights 指定有效的 .pt 文件路径，例如 my_weight/best-3.pt')
        # 诊断：如果指定文件不存在，打印提示和仓库内可用的权重列表（方便排查）
        try:
            from pathlib import Path
            import os
            if not os.path.exists(model_path):
                print(f"⚠️ 指定的权重文件不存在: {model_path}")
                pts = [str(p) for p in Path('.').rglob('*.pt')]
                if pts:
                    print('仓库中可用的 .pt 文件:', pts)
                else:
                    print('未在仓库中发现任何 .pt 权重文件')
        except Exception:
            pass
        try:
            self.model = torch.hub.load('./', 'custom', path=model_path, source='local', device=select_device)
        except Exception as e:
            # 抛出更清晰的错误提示，保留原始异常信息
            raise RuntimeError(f"加载权重失败: '{model_path}'. 请检查路径是否正确或尝试设置 --weights 指定其他权重。原始错误: {e}") from e
        self.model.conf = 0.3 # 设置一个置信度门槛


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