import os
import glob
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import torch

# 設定
CONFIG_FILE = 'aicup_config.py'
# 確保這裡選到你訓練最好的權重檔案
CHECKPOINT_FILE = 'work_dirs/aicup_config/best_coco_bbox_mAP_50.pth' 
TEST_IMG_DIR = 'data/AICUP/testing_image' # Testing Dataset 的路徑
OUTPUT_FILE = 'merged.txt'
SCORE_THR = 0.05  # 信心分數閾值，太低會雜訊多，太高會漏，建議 0.01~0.1 之間嘗試

# 初始化模型
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)

# 取得所有測試圖片
# Testing Dataset 結構: testing_image/patientXXXX/xxx.png
img_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*", "*.png"))

results_list = []

print(f"Total images to predict: {len(img_paths)}")

with open(OUTPUT_FILE, 'w') as f:
    for img_path in tqdm(img_paths):
        # 進行推論
        result = inference_detector(model, img_path)
        
        # 取得預測結果 (Tensor)
        pred_instances = result.pred_instances
        scores = pred_instances.scores
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        
        # 取得檔名 (例如 patient0051_0001)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 篩選並寫入
        for i in range(len(scores)):
            score = float(scores[i])
            if score < SCORE_THR:
                continue
            
            # 格式要求: 類別 (0)
            # PDF P.17: 圖片名稱 類別 信心分數 左上x 左上y 右下x 右下y
            # 注意 PDF 範例圖片顯示順序好像是: Name Class Score x1 y1 x2 y2
            # 但 P.17 文字描述是: 圖片名稱 -空格- 類別 ...
            # 我們依照文字描述:
            
            bbox = bboxes[i].tolist()
            x1, y1, x2, y2 = bbox
            
            # 轉整數 (PDF 寫 "整數")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 寫入一行
            line = f"{filename} 0 {score:.4f} {x1} {y1} {x2} {y2}\n"
            f.write(line)

print(f"Prediction done. Saved to {OUTPUT_FILE}")
