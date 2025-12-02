import json
import os
import glob

# ==========================================
# 設定路徑 (請確認正確)
# ==========================================
dataset_folder = "nnUNet_raw/Dataset101_Heart"
json_path = os.path.join(dataset_folder, "dataset.json")
imagesTr_dir = os.path.join(dataset_folder, "imagesTr")
labelsTr_dir = os.path.join(dataset_folder, "labelsTr")

print(f"🚀 開始修復 dataset.json...")

# 1. 掃描硬碟裡真實存在的檔案
# 搜尋所有 .nii.gz
image_files = sorted([f for f in os.listdir(imagesTr_dir) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(labelsTr_dir) if f.endswith(".nii.gz")])

print(f"📂 掃描結果：")
print(f"   imagesTr 檔案數: {len(image_files)}")
print(f"   labelsTr 檔案數: {len(label_files)}")

# 簡單檢查數量是否一致
if len(image_files) != len(label_files):
    print("❌ 警告：影像與標籤數量不符！請檢查是否有遺漏。")
    # 這裡我們取交集，確保配對成功
else:
    print("✅ 影像與標籤數量一致。")

# 2. 嘗試讀取舊 JSON 以保留 Header 資訊 (Labels 定義)
# 如果讀取失敗，我們就手動寫入標準 Header
try:
    with open(json_path, 'r') as f:
        old_data = json.load(f)
        labels = old_data.get('labels', {
            "background": 0,
            "myocardium": 1,
            "aortic_valve": 2,
            "calcification": 3
        })
        channel_names = old_data.get('channel_names', {"0": "CT"})
        print("✅ 成功讀取舊 JSON 的標籤定義。")
except:
    print("⚠️ 無法讀取舊 JSON，將使用預設標籤定義。")
    labels = {
        "background": 0,
        "myocardium": 1,
        "aortic_valve": 2,
        "calcification": 3
    }
    channel_names = {"0": "CT"}

# 3. 重新建立 training 列表
training_list = []
count = 0

for img_file in image_files:
    # 假設標籤檔名跟影像檔名一樣 (這是 nnU-Net 標準)
    # 或者是 imagesTr/case_0000.nii.gz 對應 labelsTr/case.nii.gz
    
    # 處理 _0000 後綴
    if "_0000.nii.gz" in img_file:
        label_file = img_file.replace("_0000.nii.gz", ".nii.gz")
    else:
        label_file = img_file # 假設檔名完全一樣
        
    # 檢查標籤是否存在
    if label_file in label_files:
        training_list.append({
            "image": f"./imagesTr/{img_file}",
            "label": f"./labelsTr/{label_file}"
        })
        count += 1
    else:
        print(f"❌ 找不到對應標籤：{img_file} (預期標籤: {label_file})")

# 4. 建立新的 JSON 結構
new_json = {
    "channel_names": channel_names,
    "labels": labels,
    "numTraining": len(training_list),  # 這裡絕對會是正確的數字 (55)
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
    "training": training_list
}

# 5. 寫入檔案
with open(json_path, 'w') as f:
    json.dump(new_json, f, indent=4)

print("-" * 30)
print(f"🎉 修復完成！")
print(f"修正後的 numTraining: {new_json['numTraining']}")
print(f"檔案已儲存至: {json_path}")
print("現在您可以重新執行 plan_and_preprocess 了。")