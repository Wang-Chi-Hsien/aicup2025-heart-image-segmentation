import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ==========================================
# 1. 設定路徑
# ==========================================
# 您的機率圖資料夾 (.npz)
# 請指向您目前最強模型 (例如 Best 4-Fold 集成) 的 npz 輸出位置
npz_folder = "temp_pred_fold0123_probs" 

# 您想挑選幾個做偽標籤？
TOP_K = 5 

print(f"🚀 啟動偽標籤自動篩選系統...")
print(f"來源: {npz_folder}")
print(f"目標: 挑選信心度最高的 {TOP_K} 個案例")
print("-" * 50)

# ==========================================
# 2. 評分函數
# ==========================================
def calculate_confidence_score(npz_path):
    """
    計算一張預測圖的「信心分數」。
    分數越高，代表模型越確定，品質通常越好。
    """
    try:
        # 讀取機率圖 (Channel, Z, Y, X)
        data = np.load(npz_path)['probabilities']
        
        # 1. 取得預測結果 (Argmax)
        seg = np.argmax(data, axis=0)
        
        # 2. 找出前景區域 (Label 1, 2, 3)
        # 我們只關心心臟部分的信心度，背景不重要
        foreground_mask = (seg > 0)
        
        if np.sum(foreground_mask) == 0:
            return 0.0 # 沒抓到東西，直接淘汰
            
        # 3. 取出前景的機率值
        # data.max(axis=0) 會拿到每個像素「獲勝類別」的機率值
        max_probs = data.max(axis=0)
        
        # 4. 計算前景的平均信心度
        # 例如：心肌平均確信度是 0.98 -> 分數就是 0.98
        foreground_probs = max_probs[foreground_mask]
        mean_confidence = np.mean(foreground_probs)
        
        # 5. (進階) 幾何懲罰
        # 如果心肌太小 (例如 < 5000 體素)，可能是誤判，扣分
        if np.sum(seg == 1) < 5000:
            mean_confidence *= 0.5
            
        return mean_confidence

    except Exception as e:
        print(f"Error reading {npz_path}: {e}")
        return 0.0

# ==========================================
# 3. 主迴圈
# ==========================================
scores = []
files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]

print(f"正在分析 {len(files)} 個檔案的信心度，請稍候...")

for filename in tqdm(files):
    file_path = os.path.join(npz_folder, filename)
    score = calculate_confidence_score(file_path)
    
    # 儲存結果 (分數, 檔名)
    # 檔名去掉 .npz 以便閱讀
    case_id = filename.replace('.npz', '')
    scores.append((score, case_id))

# ==========================================
# 4. 排序與選拔
# ==========================================
# 由高到低排序
scores.sort(key=lambda x: x[0], reverse=True)

print("\n🏆 篩選結果出爐 (Top Candidates)：")
print(f"{'Rank':<5} {'Case ID':<20} {'Confidence Score':<15}")
print("-" * 45)

selected_cases = []
for i in range(min(TOP_K, len(scores))):
    score, case_id = scores[i]
    print(f"{i+1:<5} {case_id:<20} {score:.5f}")
    selected_cases.append(case_id)

print("-" * 45)
print("💡 建議操作：")
print("1. 請將上述案例的 '原始影像' 和 '預測Mask' 複製到 imagesTr 和 labelsTr。")
print("2. 記得更新 dataset.json 的 numTraining 和 training 列表。")
print("3. 使用 Python 列表格式：")
print(f"selected_cases = {selected_cases}")