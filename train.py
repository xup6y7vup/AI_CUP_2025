import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- 1. 定義資料夾路徑 ---
BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "datasets"
IMG_ROOT = BASE_DIR / "training_image"
LBL_ROOT = BASE_DIR / "training_label"

# 訓練/驗證集的分割點 (同 Baseline P.34)
# patient0001~0030 為訓練集
TRAIN_PATIENTS_START = 1
TRAIN_PATIENTS_END = 45
# patient0031~0050 為驗證集
VAL_PATIENTS_START = 46
VAL_PATIENTS_END = 50

def find_patient_root(root):
    """
    (同 Baseline P.34) 
    處理 .zip 解壓縮後可能多一層資料夾的問題。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if any(d.startswith("patient") for d in dirnames):
            return Path(dirpath)
    return root # Fallback

def organize_files(start_patient, end_patient, split):
    """
    (同 Baseline P.34)
    將原始資料夾結構 (patientXXXX) 轉換為 YOLO 需要的扁平結構。
    只處理有 .txt 標註檔案的圖片。
    """
    print(f"Organizing files for '{split}' set (Patients {start_patient:04d} to {end_patient:04d})...")
    
    # 建立目標資料夾
    (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    actual_img_root = find_patient_root(IMG_ROOT)
    actual_lbl_root = find_patient_root(LBL_ROOT)

    file_moved_count = 0
    for i in range(start_patient, end_patient + 1):
        patient_id = f"patient{i:04d}"
        lbl_dir = actual_lbl_root / patient_id
        img_dir = actual_img_root / patient_id

        if not lbl_dir.exists():
            # print(f"Warning: Label directory not found {lbl_dir}, skipping.")
            continue

        # (Baseline P.34 邏輯) - "此處只採用有標註檔的圖片"
        for lbl_file in lbl_dir.glob("*.txt"):
            img_name = lbl_file.stem + ".png"
            img_path = img_dir / img_name

            if img_path.exists():
                # 複製檔案到新位置
                shutil.copy(img_path, DATASET_DIR / split / "images" / img_name)
                shutil.copy(lbl_file, DATASET_DIR / split / "labels" / lbl_file.name)
                file_moved_count += 1
            # else:
            #     print(f"Warning: Corresponding image not found {img_path}, skipping.")

    print(f"Moved {file_moved_count} image/label pairs to '{split}' set.")

def main():
    # --- 2. 整理檔案 (執行 P.34 邏輯) ---
    if not DATASET_DIR.exists():
        organize_files(TRAIN_PATIENTS_START, TRAIN_PATIENTS_END, "train")
        organize_files(VAL_PATIENTS_START, VAL_PATIENTS_END, "val")
    else:
        print(f"'{DATASET_DIR}' already exists. Skipping file organization.")

    # --- 3. 訓練模型 (P.36) - A6000 最佳化 ---
    print("Initializing A6000 (48GB VRAM) optimized training...")

    # 檢查 GPU
    print("Checking GPU...")
    if os.system("nvidia-smi") != 0:
        print("nvidia-smi command not found. Make sure NVIDIA drivers and CUDA are installed.")
        return

    # 1. 替換為最強大的 YOLOv12-X (Extra Large) 模型
    model = YOLO('yolo12x.pt')

    # 2. 使用 A6000 優化參數進行訓練
    results = model.train(
        data='aortic_valve_local.yaml',  # 我們建立的 YAML 檔
        epochs=150,                      # (A6000) 大幅增加訓練週期 (Baseline: 30)
        batch=16,                       # (A6000) 大幅增加 batch_size (Baseline: 16)
        imgsz=640,                       # (Baseline P.36) 圖片大小 640*640
        device=0,                        # 使用您的 A6000 (GPU 0)
        workers=8,                       # (A6000) 增加資料載入核心數
        patience=25,                     # 提早停止 (如果 25 個 epoch 沒進步)
        name='yolov12x_fold_1_local'  # 命名您的訓練 run
    )
    
    print("Training complete.")
    print(f"Your trained model is saved in: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()