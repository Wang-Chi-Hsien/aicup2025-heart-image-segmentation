
import numpy as np
import nibabel as nib
import os
from skimage.morphology import remove_small_objects

# ==========================================
# [修改點] 自動讀取環境變數，不再寫死路徑
# ==========================================
nnunet_raw = os.environ.get('nnUNet_raw')
if nnunet_raw is None:
    raise RuntimeError("錯誤：找不到環境變數 nnUNet_raw，請先執行 export nnUNet_raw=...")

# 設定相對於 nnUNet_raw 的路徑
dataset_dir = os.path.join(nnunet_raw, "Dataset101_Heart")
input_labels_dir = os.path.join(dataset_dir, "labelsTr") # 假設這是原始標籤
output_labels_dir = os.path.join(dataset_dir, "labelsTr_cleaned") # 建議輸出到新資料夾，再手動覆蓋

# 注意：為了安全，您可以設定直接覆蓋 labelsTr，但通常建議先備份
# 這裡依照您原本邏輯，直接覆蓋
output_labels_dir = input_labels_dir 
# ==========================================

print(f"🚀 開始執行標籤清洗...")
print(f"來源: {input_labels_dir}")
print(f"目標: {output_labels_dir}")
print("-" * 30)

for filename in os.listdir(input_labels_dir):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(input_labels_dir, filename)
        
        # 1. 讀取原始檔案
        nii = nib.load(file_path)
        data = nii.get_fdata()
        
        # 2. 複製一份數據，作為最終輸出的基底
        # 這樣做保證了 Label 2 (瓣膜) 和 Label 3 (鈣化) 絕對是原始狀態
        final_data = data.copy()
        
        # 3. 提取心肌 mask (Label 1)
        myo_mask = (data == 1).astype(bool)
        
        # 計算處理前的體積 (用於 Log)
        orig_vol = np.sum(myo_mask)
        
        if orig_vol > 0:
            # === 核心操作：僅移除小物件 ===
            # min_size=100: 小於 100 個體素的孤立點會被刪除
            # 這在 3D 空間中是很小的雜訊，但足以保留主要結構
            cleaned_mask = remove_small_objects(myo_mask, min_size=100)
            
            # === 將清洗後的心肌寫回 final_data ===
            # A. 先將 final_data 中原本是 1 的位置歸零
            final_data[final_data == 1] = 0
            
            # B. 填入清洗後的心肌 (設為 1)
            final_data[cleaned_mask] = 1
            
            # Log 顯示刪除了多少噪點
            diff = orig_vol - np.sum(cleaned_mask)
            if diff > 0:
                print(f"[{filename}] 清除了 {diff} 個孤立噪點 (Label 1)")
            else:
                print(f"[{filename}] Label 1 很乾淨，未變動")
                
        else:
            print(f"[{filename}] ⚠️ 警告：此案例沒有心肌 Label 1")

        # 4. 關鍵修復：強制轉為整數 (uint8)
        # 這解決了之前的 Crash 問題
        final_data = np.round(final_data).astype(np.uint8)
        
        # 5. 儲存
        # 使用原始 Header 但更新資料類型
        new_header = nii.header.copy()
        new_header.set_data_dtype(np.uint8)
        
        new_nii = nib.Nifti1Image(final_data, nii.affine, new_header)
        target_path = os.path.join(output_labels_dir, filename)
        nib.save(new_nii, target_path)

print("-" * 30)
print("✅ 最終版清洗完成！")
print("1. Label 1 (心肌)：已去除孤立噪點，邊緣未縮小，結構未斷裂。")
print("2. Label 2 & 3：保證 100% 原汁原味。")
print("3. 資料格式：已修正為 uint8。")
print("\n接下來請執行：")
print("nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity")