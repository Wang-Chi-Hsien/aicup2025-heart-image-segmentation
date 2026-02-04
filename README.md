# AICUP 2025 Heart Segmentation Challenge - Final Solution (SOTA)

This repository contains the source code and solution for the **AICUP 2025 Heart CT Image Segmentation Challenge**.
Our approach is built upon the **nnU-Net v2** framework, integrating **Label Cleaning**, **Semi-Supervised Learning (Pseudo-labeling)**, and **Probability Boosting Ensemble** strategies. These methods effectively address challenges related to small datasets and class imbalance.

---

## 1. Environment Setup

### Hardware Requirements
* **OS:** Linux (Ubuntu 20.04/22.04 recommended)
* **Python:** 3.10+
* **GPU:** NVIDIA RTX 3090 / 4090 / 5090 (Recommended VRAM $\ge$ 24GB)

### Installation Steps
1.  **Install Basic Packages**:
    ```bash
    # Install PyTorch (Choose the command appropriate for your CUDA version)
    pip install torch torchvision torchaudio

    # Install nnU-Net v2 and other dependencies
    pip install nnunetv2
    pip install numpy nibabel scikit-image tqdm
    ```

2.  **Set nnU-Net Environment Variables**:
    Execute the following commands in your terminal (or add them to your `~/.bashrc`) to define the data storage paths:
    ```bash
    export nnUNet_raw="/path/to/your/nnUNet_raw"
    export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
    export nnUNet_results="/path/to/your/nnUNet_results"
    ```

---

## 2. File Structure

Please organize the code as follows:

```text
.
├── README.md                   # This documentation
├── src/
│   ├── preprocess/
│   │   ├── clean_labels.py     # Label Cleaning (Morphological noise removal)
│   │   ├── select_pseudo.py    # Filter high-confidence pseudo-labels
│   │   └── move_pseudo.py      # Move pseudo-labels to the training set
│   ├── inference/
│   │   └── ensemble.py         # Weighted Ensemble + Probability Boosting core script
│   ├── postprocess/
│   │   ├── post_process.py     # Final Post-processing (Header fix, Largest Component)
│   │   └── pack_submission.py  # Pack files for submission
│   └── utils/
│       └── fix_dataset_json.py # Tool to automatically generate dataset.json
└── scripts/                    # (Optional) Place .sh training scripts here
```

## 3. Data Preparation & Structure
⚠️ Important Note: This project uses the nnU-Net framework, so the dataset must strictly adhere to nnU-Net format requirements (Dataset Fingerprint). Before starting training, please ensure your data conforms to the following structure:

### 3.1 Directory Structure
Please create three folders on your disk and point the environment variables to them. nnUNet_raw must contain the raw data, while nnUNet_preprocessed and nnUNet_results should remain Empty, as the program will automatically write data to them.

```text
(Root Directory)
├── nnUNet_raw/               <-- [User Input] Manually place data here
│   └── Dataset101_Heart/
│       ├── dataset.json
│       ├── imagesTr/
│       ├── labelsTr/
│       └── imagesTs/
│
├── nnUNet_preprocessed/      <-- [Auto-Generated] Keep empty, preprocessing scripts output here
│
└── nnUNet_results/           <-- [Auto-Generated] Keep empty, training scripts save weights here
```
Please set the path pointed to by the nnUNet_raw environment variable as follows:

```text
nnUNet_raw/
└── Dataset101_Heart/
    ├── dataset.json          <-- (Must contain correct metadata)
    ├── imagesTr/             <-- (Training Images)
    │   ├── patient0001_0000.nii.gz  <-- Filename must include the 4-digit suffix _0000
    │   ├── patient0002_0000.nii.gz
    │   └── ...
    ├── labelsTr/             <-- (Training Labels)
    │   ├── patient0001.nii.gz       <-- Filename must NOT include _0000
    │   ├── patient0002.nii.gz
    │   └── ...
    └── imagesTs/             <-- (Test Images)
        ├── patient0051_0000.nii.gz
        └── ...
```
### 3.2 Key Naming Convention
Images: Must end with .nii.gz and must include the modality identifier _0000 (e.g., case_001_0000.nii.gz).

Labels: Must end with .nii.gz and must not include _0000 (e.g., case_001.nii.gz).

### 3.3 Generating dataset.json
If your environment does not have a dataset.json, please run our provided utility script to automatically generate it based on existing files:

```bash
# Ensure imagesTr and labelsTr files are in place
python src/utils/fix_dataset_json.py
# (Note: Please check that file paths inside fix_dataset_json.py are correct)
```
---
## 4. Training Pipeline

### Step 1: Data Preprocessing
Since the original data's Label 1 (Myocardium) contains some isolated noise, we first perform cleaning.
```bash
# Clean raw labels (Remove isolated spots < 100 voxels)
python src/preprocess/clean_labels.py

# Execute nnU-Net standard preprocessing
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity
```

### Step 2: Base Model Training
1. **Train basic 3D U-Net models**
    Based on cross-validation results, we keep the best performing Folds 0, 2, and 3.

    ```bash
    nnUNetv2_train 101 3d_fullres 0 --npz
    nnUNetv2_train 101 3d_fullres 2 --npz
    nnUNetv2_train 101 3d_fullres 3 --npz
    ```


2. **Training 2D Model**
    To capture sharp edges between Z-axis slices, we additionally trained a 2D U-Net. 
    Since we found Fold 2 performed the best, we only trained the 2D model on Fold 2.

    1. **Run preprocessing for 2D configuration** (If not run previously):
        ```bash
        nnUNetv2_plan_and_preprocess -d 101 -c 2d --verify_dataset_integrity
        ```

    2. **Execute Training (Fold 2):**：
        ```bash
        nnUNetv2_train 101 2d 2 --npz
        ```


### Step 3: Pseudo-labeling
To enhance the model's generalization capability, we use the test set for semi-supervised learning:

1. Use the strongest base model (Fold 2) to predict on the test set.
2. Run the selection script to automatically pick the top 5 data entries with the highest Confidence Score:
    ```bash
    python src/preprocess/select_pseudo.py
    ```
3.  將這 5 筆資料搬運至訓練集並更新 `dataset.json`：
    ```bash
    python src/preprocess/move_pseudo.py
    ## (You can use fix_dataset_json.py again to auto-update dataset.json)
    ```
4.  **Retrain Fold 1** (Using the enhanced dataset):
    ```bash
    # Must run preprocessing again to update fingerprints
    nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity
    
    # Train enhanced Fold 1
    nnUNetv2_train 101 3d_fullres 1 --npz
    ```

### Step 4: Inference & Ensemble
This is the key step to improving the score. We combine "Old Models" and "New Models" and optimize probabilities for small objects.

1.  **Generate Probability Maps (.npz)**：
    Ensure the --save_probabilities parameter is added when running predictions for all models.
    ```bash
    # Example commands
    nnUNetv2_predict -i ... -o temp_pred_old_f023 -f 0 2 3 --save_probabilities
    nnUNetv2_predict -i ... -o temp_pred_new_f1 -f 1 --save_probabilities
    nnUNetv2_predict -i ... -o predictions_2d_raw -f 1 --save_probabilities
    ```

2.  **Execute Weighted Ensemble**：
    Run the following script to merge the two sets of probability maps:
    ```bash
    python src/inference/ensemble.py
    ```

### Step 5: Post-processing & Packing
Finally, perform morphological post-processing and NIfTI Header repair, then pack for submission.
```bash
python src/postprocess/post_process.py
python src/postprocess/pack_submission.py
```

---

## 5. Parameter Settings

This solution includes the following key adjustments tailored to the specific characteristics of this task:

### A. Ensemble Weights
We adopt a "Strong + Strong" strategy but give higher stability weight to the verified old models.
*   **Group Old (Folds 0, 2, 3)**: Weights **3.2** (Representing the majority vote base).
*   **Group New (Fold 1)**: Weights **1.0** (Introducing new features from pseudo-labels, but not dominating the decision).

### B. Probability Boosting
For extremely small targets (Aortic Valve, Calcification), the model tends to be conservative (predicting as background). We calibrate at the Softmax probability layer:
*   **Label 2 (Aortic Valve)**: Probability value $\times$ **1.15**
*   **Label 3 (Calcification)**: Probability value $\times$ **1.15**
*   **Goal**： Lower the decision threshold for small objects, significantly improving Recall.
### C. Post-processing
*   **Keep Largest Component**: For Label 1 (Myocardium), only keep the largest connected region to remove spraying noise.
*   **Geometric Repair**: Disable dilation and closing operations; only preserve original structures to avoid misjudgments.

---

## 6. 致謝 (Credits)
This project uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as the core framework.
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

# AICUP 2025 Heart Segmentation Challenge - 決賽解決方案 (SOTA Solution)

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
│   ├── postprocess/
│   │    ├── post_process.py     # 最終後處理 (Header修復、最大連通組件)
│   │    └── pack_submission.py  # 打包上傳檔案
│   └── utils/
│       └── fix_dataset_json.py  # 自動生成 dataset.json 工具
└── scripts/                    # (可選) 放置 .sh 訓練腳本
```

---

## 3. 資料準備與格式要求 (Data Preparation & Structure)

⚠️ **重要提示 (Important Note)**:
本專案使用 nnU-Net 框架，因此資料集必須嚴格遵守 nnU-Net 的格式要求 (Dataset Fingerprint)。在開始訓練之前，請確保您的資料符合以下結構：

### 3.1 資料夾結構 (Directory Structure)
請在您的硬碟中建立三個資料夾，並將環境變數指向它們。
其中 `nnUNet_raw` 必須包含原始資料，而 `nnUNet_preprocessed` 與 `nnUNet_results` 請保持**為空 (Empty)**，程式將自動寫入資料。

```text
(Root Directory)
├── nnUNet_raw/               <-- [User Input] 需手動放入資料
│   └── Dataset101_Heart/
│       ├── dataset.json
│       ├── imagesTr/
│       ├── labelsTr/
│       └── imagesTs/
│
├── nnUNet_preprocessed/      <-- [Auto-Generated] 請留空，預處理腳本會自動輸出至此
│
└── nnUNet_results/           <-- [Auto-Generated] 請留空，訓練腳本會將模型權重存於此
```

請將 `nnUNet_raw` 環境變數指向的路徑設定如下：

```text
nnUNet_raw/
└── Dataset101_Heart/
    ├── dataset.json          <-- (必須包含正確的 metadata)
    ├── imagesTr/             <-- (訓練影像 Training Images)
    │   ├── patient0001_0000.nii.gz  <-- 檔名必須包含四位數後綴 _0000
    │   ├── patient0002_0000.nii.gz
    │   └── ...
    ├── labelsTr/             <-- (訓練標籤 Training Labels)
    │   ├── patient0001.nii.gz       <-- 檔名不含 _0000
    │   ├── patient0002.nii.gz
    │   └── ...
    └── imagesTs/             <-- (測試影像 Test Images)
        ├── patient0051_0000.nii.gz
        └── ...
```

### 3.2 關鍵命名規則 (Naming Convention)
*   **影像檔 (Images)**：必須以 `.nii.gz` 結尾，且**必須**包含模態標識符 `_0000` (例如 `case_001_0000.nii.gz`)。
*   **標籤檔 (Labels)**：必須以 `.nii.gz` 結尾，且**不可**包含 `_0000` (例如 `case_001.nii.gz`)。

### 3.3 自動生成 dataset.json (Generating dataset.json)
若您的環境中沒有 `dataset.json`，請執行我們提供的輔助腳本來根據現有檔案自動生成：
```bash
# 請確保 imagesTr 與 labelsTr 檔案已就位
python src/utils/fix_dataset_json.py
# (注意：請確認 fix_dataset_json.py 內的檔案路徑設定正確)
```

---

## 4. 訓練流程與執行步驟 (Training Pipeline)

### 步驟一：資料前處理 (Data Preprocessing)
由於原始資料的 Label 1 (Myocardium) 存在部分孤立噪點，我們先執行清洗。
```bash
# 清洗原始標籤 (去除 < 100 體素的孤立點)
python src/preprocess/clean_labels.py

# 執行 nnU-Net 標準預處理
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity
```

### 步驟二：基底模型訓練 (Base Model Training)
1. **訓練基礎的 3D U-Net 模型**
    根據交叉驗證結果，我們保留表現最好的 Fold 0, 2, 3。
    ```bash
    nnUNetv2_train 101 3d_fullres 0 --npz
    nnUNetv2_train 101 3d_fullres 2 --npz
    nnUNetv2_train 101 3d_fullres 3 --npz
    ```


2. **訓練 2D 模型 (Training 2D Model)**
    為了捕捉 Z 軸切片間的銳利邊緣，我們額外訓練了一個 2D U-Net。
    由於我們發現 Fold 2 的表現最好，因此只針對 Fold 2 進行 2D 訓練。

    1. **執行 2D 配置的預處理** (若之前未執行過)：
        ```bash
        nnUNetv2_plan_and_preprocess -d 101 -c 2d --verify_dataset_integrity
        ```

    2. **執行訓練 (Fold 2)**：
        ```bash
        nnUNetv2_train 101 2d 2 --npz
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
    # (可再次使用 fix_dataset_json.py 自動更新 dataset.json)
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
    nnUNetv2_predict -i ... -o predictions_2d_raw -f 1 --save_probabilities
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

## 5. 參數設定說明 (Parameter Settings)

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

## 6. 致謝 (Credits)
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
