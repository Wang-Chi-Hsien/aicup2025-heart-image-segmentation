import os
import shutil
import nibabel as nib
import numpy as np
from skimage.measure import label

# --- 設定路徑 (已修改為相對路徑) ---
# 輸入：nnU-Net 預測結果 (或是集成後的結果) 的輸出路徑
# 使用相對路徑，預設在專案根目錄下
prediction_folder = "./submission_optimized_final"

# 輸出：處理完後要存放的路徑
postprocessed_folder = "./postprocessed_final_ensemble"

if not os.path.exists(postprocessed_folder):
    os.makedirs(postprocessed_folder)

def keep_largest_connected_component(volume, labels_info):
    """
    對多標籤的 volume，針對每個標籤獨立保留最大連通物件
    """
    final_volume = np.zeros_like(volume, dtype=volume.dtype)
    
    for label_val, label_name in labels_info.items():
        if label_val == 0: # 跳過背景
            continue
            
        mask = (volume == label_val)
        if np.sum(mask) == 0: # 如果這個標籤不存在，跳過
            continue
        
        # 找出所有連通元件
        labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
        
        if num_features > 0:
            # 計算每個元件的大小
            component_sizes = np.bincount(labeled_mask.ravel())[1:] # 忽略背景0
            largest_component_label = np.argmax(component_sizes) + 1
            
            # 只保留最大的元件
            cleaned_mask = (labeled_mask == largest_component_label)
            final_volume[cleaned_mask] = label_val
            
    return final_volume

# 從 dataset.json 讀取標籤資訊
labels_info = {1: "myocardium", 2: "aortic_valve", 3: "calcification"}

print(f"🚀 開始執行後處理...")
print(f"來源: {prediction_folder}")
print(f"目標: {postprocessed_folder}")

for filename in os.listdir(prediction_folder):
    if filename.endswith(".nii.gz"):
        print(f"Processing {filename}...")
        
        # 載入預測檔案
        nifti_path = os.path.join(prediction_folder, filename)
        nifti = nib.load(nifti_path)
        volume = nifti.get_fdata()
        
        # 執行後處理
        cleaned_volume = keep_largest_connected_component(volume, labels_info)
        
        # 儲存處理後的檔案
        output_nifti = nib.Nifti1Image(cleaned_volume.astype(np.uint8), nifti.affine, nifti.header)
        nib.save(output_nifti, os.path.join(postprocessed_folder, filename))
        
print("✅ Post-processing finished!")