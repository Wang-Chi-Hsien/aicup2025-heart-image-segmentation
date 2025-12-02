---

# AICUP 2025 Heart Segmentation Challenge

本專案為 **AICUP 2025 心臟電腦斷層掃描影像分割競賽** 的原始程式碼與解決方案。
我們的方法基於 **nnU-Net v2** 框架，並結合了 **標籤清洗 (Label Cleaning)**、**半監督學習 (Pseudo-labeling)** 以及 **機率灌水集成 (Probability Boosting Ensemble)** 策略，有效解決了小數據集與類別不平衡的問題。

---

## 1. 環境建置 (Environment Setup)

### 硬體需求
*   OS: Linux (Ubuntu 20.04/22.04 推薦)
*   Python: 3.10+
*   GPU: NVIDIA RTX 3090 / 4090 / 5090 (建議 VRAM >= 24GB)

### 安裝步驟
1.  **安裝基礎套件**：
    ```bash
    # 安裝 PyTorch (請依據您的 CUDA 版本選擇對應指令)
    pip install torch torchvision torchaudio

    # 安裝 nnU-Net v2 及其他依賴
    pip install nnunetv2
    pip install numpy nibabel scikit-image tqdm
    ```

2.  **設定 nnU-Net 環境變數**：
    請在終端機執行以下指令（或寫入 `~/.bashrc`），設定資料存放路徑：
    ```bash
    export nnUNet_raw="/path/to/your/nnUNet_raw"
    export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
    export nnUNet_results="/path/to/your/nnUNet_results"
    ```

---

## 2. 檔案結構 (File Structure)

請將程式碼整理如下：

```text
.
├── README.md                   # 本說明文件
├── src/
│   ├── preprocess/
│   │   ├── clean_labels.py     # 標籤清洗 (形態學去除噪點)
│   │   ├── select_pseudo.py    # 篩選高信心度的偽標籤
│   │   └── move_pseudo.py      # 搬運偽標籤至訓練集
│   ├── inference/
│   │   └── ensemble.py         # 加權集成 + 機率灌水核心腳本
│   └── postprocess/
│       ├── post_process.py     # 最終後處理 (Header修復、最大連通組件)
│       └── pack_submission.py  # 打包上傳檔案
└── scripts/                    # (可選) 放置 .sh 訓練腳本
```

---

## 3. 訓練流程與執行步驟 (Training Pipeline)

### 步驟一：資料前處理 (Data Preprocessing)
由於原始資料的 Label 1 (Myocardium) 存在部分孤立噪點，我們先執行清洗。
```bash
# 清洗原始標籤 (去除 < 100 體素的孤立點)
python src/preprocess/clean_labels.py

# 執行 nnU-Net 標準預處理
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity
```

### 步驟二：基底模型訓練 (Base Model Training)
訓練基礎的 3D U-Net 模型。根據交叉驗證結果，我們保留表現最好的 Fold 0, 2, 3。
```bash
nnUNetv2_train 101 3d_fullres 0 --npz
nnUNetv2_train 101 3d_fullres 2 --npz
nnUNetv2_train 101 3d_fullres 3 --npz
```

### 步驟三：偽標籤強化 (Pseudo-labeling)
為了增強模型的泛化能力，我們利用測試集進行半監督學習：
1.  使用最強的基底模型 (Fold 2) 對測試集進行預測。
2.  執行篩選腳本，自動挑選信心度 (Confidence Score) 最高的 5 筆資料：
    ```bash
    python src/preprocess/select_pseudo.py
    ```
3.  將這 5 筆資料搬運至訓練集並更新 `dataset.json`：
    ```bash
    python src/preprocess/move_pseudo.py
    # (請手動或使用腳本更新 dataset.json 的 numTraining 為 55)
    ```
4.  **重新訓練 Fold 1** (使用強化後的數據集)：
    ```bash
    # 需重新跑一次預處理以更新指紋
    nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity
    
    # 訓練強化版 Fold 1
    nnUNetv2_train 101 3d_fullres 1 --npz
    ```

### 步驟四：預測與集成 (Inference & Ensemble)
這是提升分數的關鍵步驟。我們結合了「舊模型」與「新模型」，並針對小物件進行機率優化。

1.  **生成機率圖 (.npz)**：
    確保對所有模型執行預測時加上 `--save_probabilities` 參數。
    ```bash
    # 範例指令
    nnUNetv2_predict -i ... -o temp_pred_old_f023 -f 0 2 3 --save_probabilities
    nnUNetv2_predict -i ... -o temp_pred_new_f1 -f 1 --save_probabilities
    ```

2.  **執行加權集成 (Weighted Ensemble)**：
    執行以下腳本將兩組機率圖合併：
    ```bash
    python src/inference/ensemble.py
    ```

### 步驟五：後處理與打包 (Post-processing)
最後執行形態學後處理與 NIfTI Header 修復，並打包提交。
```bash
python src/postprocess/post_process.py
python src/postprocess/pack_submission.py
```

---

## 4. 參數設定說明 (Parameter Settings)

本方案針對本次任務特性進行了以下關鍵調整：

### A. 集成權重 (Ensemble Weights)
我們採用「強強聯手」的策略，但給予經過驗證的舊模型較高的穩定性權重。
*   **Group Old (Folds 0, 2, 3)**: 權重 **3.2** (代表多數決基礎)。
*   **Group New (Fold 1)**: 權重 **1.0** (引入偽標籤的新特徵，但不至於主導決策)。

### B. 機率灌水 (Probability Boosting)
針對極小目標（主動脈瓣膜、鈣化），模型傾向於保守（預測為背景）。我們在 Softmax 機率層進行校準：
*   **Label 2 (Aortic Valve)**: 機率值 $\times$ **1.15**
*   **Label 3 (Calcification)**: 機率值 $\times$ **1.15**
*   **目的**：降低小物件的決策門檻，顯著提升 Recall (召回率)。

### C. 後處理 (Post-processing)
*   **Keep Largest Component**: 針對 Label 1 (心肌)，只保留最大連通區域，移除噴灑狀 (Spraying) 噪點。
*   **幾何修復**: 關閉膨脹與閉運算，僅保留原始結構以避免誤判。

---

## 5. 致謝 (Credits)
本專案使用了 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 作為核心框架。
```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```