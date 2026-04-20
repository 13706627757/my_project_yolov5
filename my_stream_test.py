import torch
import cv2
import os
import time

def main():
    # ==========================================
    # 【加载区】这里的代码只会运行一次！(也就是只需要等一次 1 分钟)
    # ==========================================
    print("🚀 正在初始化系统，请稍候...")
    model = torch.hub.load('./', 'custom', path='yolov5s.pt', source='local')
    
    # 开发调试用 CPU，实战演示请改成 '0' (GPU)
    model.device = 'cpu'  
    model.conf = 0.25 
    print("✅ 模型加载完毕！系统进入实时待命状态。\n")

    output_dir = 'result_test_img'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # 【循环区】死循环，只要不按 Ctrl+C 就会一直跑
    # ==========================================
    # 假设 0 是你的 USB 摄像头。如果你目前没接摄像头，咱们可以用连续读图来模拟！
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("❌ 找不到摄像头！请检查 USB 连接。")
        return

    frame_count = 0
    while True:
        # 1. 抓取当前画面
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取画面。")
            break

        frame_count += 1
        
        # 2. 瞬间推理！(因为模型已经在内存/显存里了，这步极快)
        results = model(frame)

        # 3. 提取数据并执行你的“报警逻辑”
        predictions = results.pandas().xyxy[0]
        
        # 假设我们只关心画面里有没有出现人
        person_detected = False
        for index, row in predictions.iterrows():
            name = row['name']
            xmin = int(row['xmin'])
            
            if name == 'person':
                person_detected = True
                print(f"[第{frame_count}帧] ⚠️ 警告：发现人员！X坐标: {xmin}")
                
                # 【你的毕设逻辑】
                if xmin > 300:
                    print("   -> 🚨 触发报警！电机开始转动！")
        
        # 4. 智能保存 (只有当画面里有人的时候，才把图片存下来当证据，节省 SD 卡空间)
        if person_detected:
            rendered_img = results.render()[0]
            save_path = os.path.join(output_dir, f"alert_{int(time.time())}.jpg")
            cv2.imwrite(save_path, rendered_img)
            
            # 为了防止存得太快把硬盘撑爆，发现人之后稍微睡个 1 秒
            time.sleep(1)

    # 释放资源
    cap.release()

if __name__ == "__main__":
    main()