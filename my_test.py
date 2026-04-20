import torch
import cv2
import os
import time
# ==========================================
# 1. 初始化模型 (加载本地的 YOLOv5 目录和权重)
# ==========================================
print("正在加载模型...")
# 参数解释：
# './' : 代表使用当前目录的 yolov5 代码，而不是去 github 下载
# 'custom' : 表示我们要用自己的权重文件
# path : 你的权重文件路径
# source='local' : 强制使用本地代码，防止因为断网报错
model = torch.hub.load('./', 'custom', path='yolov5s.pt', source='local')

# 设置运行设备：刚才配好的 GPU 填 '0'，测试填 'cpu'
model.device = 'cpu'  # 你可以改成 '0' 试试全速运行
# 设置检测阈值（置信度低于 0.25 的不要）
model.conf = 0.25 

# ==========================================
# 2. 读取图像并推理
# ==========================================
img_path = 'data/images/zidane.jpg'
img = cv2.imread(img_path)  # 用 OpenCV 读取图片

print("开始检测...")
# 推理：就这一句话，直接把图片扔给模型！
results = model(img)

# ==========================================
# 3. 提取你要的数据（重点！为你的其他功能做准备）
# ==========================================
print("\n=== 检测结果数据 ===")
# results.pandas().xyxy[0] 会返回一个包含所有检测框信息的表格
predictions = results.pandas().xyxy[0]
print(predictions)

# 举个例子：如果你想根据检测结果做判断
for index, row in predictions.iterrows():
    name = row['name']       # 类别名称，比如 'person'
    conf = row['confidence'] # 置信度
    xmin = int(row['xmin'])  # 框的左上角 x 坐标
    
    print(f"-> 发现目标: {name}, 坐标x: {xmin}, 可靠度: {conf:.2f}")
    
    # 【这里就是你的毕设逻辑切入点！】
    # if name == 'person' and xmin > 300:
    #     print("有人越界！触发报警电机转动...")

# ==========================================
# 4. 可视化保存（把画好框的图存下来看看）
# ==========================================
# results.render() 会在原图上画好框，我们把它取出来
rendered_img = results.render()[0]

# 使用你刚才决定的文件夹名称
output_dir = 'result_test_img'

# 安全检查：如果文件夹因为某种原因不在，代码会自动建一个，防止报错
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成一个带当前时间戳的图片名，例如 result_16382012.jpg
file_name = f"result_{int(time.time())}.jpg"

# 拼接完整的保存路径
save_path = os.path.join(output_dir, file_name)

# 保存图片！
cv2.imwrite(save_path, rendered_img)
print(f"\n✅ 帅！结果图片已成功保存至: {save_path}")