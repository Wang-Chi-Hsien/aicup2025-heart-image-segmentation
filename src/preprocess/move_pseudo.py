import os
import shutil

# ==========================================
# 1. 關鍵設定 (請確認您的路徑!)
# ==========================================
# 來源 A: 原始測試影像 (imagesTs)
test_images_dir = "nnUNet_raw/Dataset101_Heart/imagesTs"

# 來源 B: 您的最佳預測結果 (Mask)
# ⚠️ 請改成您分數最高、做過後處理的那個資料夾名稱！
# 例如: submission_final_safe 或 output_best4_postprocessed
best_prediction_dir = "output_fold0123_fixed_postprocessed" 

# 目標位置 (訓練集)
target_imagesTr = "nnUNet_raw/Dataset101_Heart/imagesTr"
target_labelsTr = "nnUNet_raw/Dataset101_Heart/labelsTr"

# 您的 Top 5 名單 (已幫您填好)
selected_cases = ['patient0057', 'patient0097', 'patient0061', 'patient0084', 'patient0094']

print(f"🚀 開始搬運 5 筆偽標籤資料...")
print(f"影像來源: {test_images_dir}")
print(f"標籤來源: {best_prediction_dir}")

for case_id in selected_cases:
    # 定義來源檔名
    # 影像通常有 _0000，預測檔通常沒有
    src_img = os.path.join(test_images_dir, f"{case_id}_0000.nii.gz")
    src_seg = os.path.join(best_prediction_dir, f"{case_id}.nii.gz")
    
    # 定義目標檔名 (加上 pseudo_ 前綴)
    dst_img = os.path.join(target_imagesTr, f"pseudo_{case_id}_0000.nii.gz")
    dst_seg = os.path.join(target_labelsTr, f"pseudo_{case_id}.nii.gz")
    
    # 執行複製
    if os.path.exists(src_img) and os.path.exists(src_seg):
        shutil.copy(src_img, dst_img)
        shutil.copy(src_seg, dst_seg)
        print(f"✅ [成功] pseudo_{case_id} 已加入訓練集")
    else:
        print(f"❌ [失敗] 找不到檔案: {case_id}")
        if not os.path.exists(src_img): print(f"   - 缺影像: {src_img}")
        if not os.path.exists(src_seg): print(f"   - 缺標籤: {src_seg}")

print("搬運完成！")