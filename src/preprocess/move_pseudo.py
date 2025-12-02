import os
import shutil

# ==========================================
# [修改點] 混合使用 環境變數 + 相對路徑
# ==========================================
# 1. 取得 nnU-Net 資料路徑
nnunet_raw = os.environ.get('nnUNet_raw')
if nnunet_raw is None:
    raise RuntimeError("請先設定 export nnUNet_raw=...")

test_images_dir = os.path.join(nnunet_raw, "Dataset101_Heart/imagesTs")
target_imagesTr = os.path.join(nnunet_raw, "Dataset101_Heart/imagesTr")
target_labelsTr = os.path.join(nnunet_raw, "Dataset101_Heart/labelsTr")

# 2. 設定預測結果來源 (相對路徑)
# 這裡指向主辦方剛剛生成的、做過後處理的預測結果
best_prediction_dir = "./submission_temp_for_pseudo" 

# 3. 為了確保重現性，將 ID 寫死 (Hard-code)
# 這樣主辦方不需要重新篩選，直接用這 5 個最好的
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