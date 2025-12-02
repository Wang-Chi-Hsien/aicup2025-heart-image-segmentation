import os
import shutil
import zipfile

# --- 設定路徑 ---
# 後處理完的資料夾
source_folder = "/home/temp/Luke/AICUP_Heart/postprocessed_final_ensemble"
# 準備要壓縮的資料夾
submission_folder = "./submission_temp/"
# 最終 zip 檔案
zip_filename = "predict.zip"

if os.path.exists(submission_folder):
    shutil.rmtree(submission_folder)
os.makedirs(submission_folder)

# 重新命名檔案以符合競賽要求 (e.g., patient0051.nii.gz -> patient0051_predict.nii.gz)
for filename in os.listdir(source_folder):
    if filename.endswith(".nii.gz"):
        # nnU-Net 的輸出檔名就是 patientXXXX.nii.gz，我們需要加上 _predict
        base_name = filename.split('.')[0]
        new_name = f"{base_name}_predict.nii.gz"
        
        shutil.copyfile(os.path.join(source_folder, filename), 
                        os.path.join(submission_folder, new_name))

print("Files renamed and copied for submission.")

# 打包成 predict.zip
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk(submission_folder):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)

print(f"Submission file '{zip_filename}' created successfully!")

# 清理臨時資料夾
shutil.rmtree(submission_folder)
