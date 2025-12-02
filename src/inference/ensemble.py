import numpy as np
import nibabel as nib
import os

# ==============================
# 1. 設定路徑
# ==============================
folder_group_main = "temp_pred_old_f023" # 內含 F0, F2, F3 (權重已由 nnU-Net 平均過)
folder_new_1 = "temp_pred_new_f1"        # New F1
folder_2d = "temp_pred_2d_fold2"         # 2D Model

# 輸出
output_folder = "submission_optimized_final"
raw_images_folder = "nnUNet_raw/Dataset101_Heart/imagesTs" # 用來修 Header

os.makedirs(output_folder, exist_ok=True)

# ==============================
# 2. 超參數微調 (關鍵!)
# ==============================
# Group Main 代表 3 個模型。
# 我們假設 Fold 2 在裡面，所以給這個群組較高權重
w_main = 3.2  

# New F1 代表 1 個模型
w_new1 = 1.0  

# 2D 模型 (給極低權重，當作 Tie-breaker)
# 如果之前加了變差，試試 0.1 或 0.05，或者乾脆設 0 關掉它
w_2d = 0 

# 類別機率灌水 (Boosting)
# 1.0 = 不變, 1.1 = 增加 10% 機率
boost_valve = 1.15 
boost_calc = 1.15

print(f"🚀 啟動終極優化集成...")
print(f"權重: Main({w_main}) + New1({w_new1}) + 2D({w_2d})")
print(f"灌水: Valve(x{boost_valve}), Calc(x{boost_calc})")

files = [f for f in os.listdir(folder_group_main) if f.endswith('.npz')]

for filename in files:
    # 1. 讀取數據
    # Group Main (F0, F2, F3)
    data_main = np.load(os.path.join(folder_group_main, filename))['probabilities']
    
    # New F1
    path_1 = os.path.join(folder_new_1, filename)
    if os.path.exists(path_1):
        data_1 = np.load(path_1)['probabilities']
    else:
        data_1 = np.zeros_like(data_main)
        w_new1_act = 0
    
    # 2D Model
    path_2d = os.path.join(folder_2d, filename)
    if os.path.exists(path_2d):
        data_2d = np.load(path_2d)['probabilities']
    else:
        data_2d = np.zeros_like(data_main)
        w_2d_act = 0

    # 2. 加權平均
    final_probs = (data_main * w_main) + (data_1 * w_new1) + (data_2d * w_2d)
    
    # 3. 執行機率灌水 (Boosting)
    # channel 0:BG, 1:Myo, 2:Valve, 3:Calc
    final_probs[2] *= boost_valve
    final_probs[3] *= boost_calc

    # 4. 轉回 Mask
    seg_mask = np.argmax(final_probs, axis=0).astype(np.uint8)

    # 5. 存檔與修復
    nii_filename = filename.replace('.npz', '.nii.gz')
    raw_filename = filename.replace('.npz', '_0000.nii.gz')
    
    # 嘗試找原始檔以修復形狀
    raw_path = os.path.join(raw_images_folder, raw_filename)
    if not os.path.exists(raw_path): raw_path = os.path.join(raw_images_folder, nii_filename)

    if os.path.exists(raw_path):
        ref_nii = nib.load(raw_path)
        # 自動轉置
        if seg_mask.shape != ref_nii.shape:
            seg_mask = seg_mask.transpose(2, 1, 0)
        
        # 安全存檔
        new_nii = nib.Nifti1Image(seg_mask, ref_nii.affine)
        new_nii.header.set_xyzt_units(2)
        nib.save(new_nii, os.path.join(output_folder, nii_filename))
    else:
        print(f"❌ Header 遺失: {filename}")

print("✅ 優化完成！請務必執行 SOTA 後處理 (post_process_final.py) 再提交！")