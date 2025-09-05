# Large-scale Energy Anomaly Detection in Commercial Buildings (LEAD)

This project focuses on detecting abnormal energy consumption patterns in commercial buildings using advanced **machine learning techniques**. The main goal is to highlight **data preprocessing, handling imbalanced data and applying modern ML models (XGBoost, DNN, Logistic Regression)** to achieve high performance in anomaly detection.

---

## 1. Project Overview
- Buildings account for approximately one-third of global energy consumption, with ~20% wasted due to inefficiencies (equipment malfunctions, misconfigurations, aging systems, human errors).  
- Detecting anomalies in energy consumption can help optimize energy usage and reduce waste.  
- This project emphasizes **large-scale data preprocessing, feature engineering, and machine learning modeling** for anomaly detection.

---

## 2. Processed Data & Files

- **Source:** ASHRAE Great Energy Predictor Competition (2016)  
- **Original data:** 400 commercial buildings  
- **Processed training data:** 201 buildings (~1.75M records)  

The **processed dataset `train_meta_label.csv`** is the result of merging raw data `train.csv` with `building_metadata.csv` and applying preprocessing steps to make it ready for model training.

### Files

**train.csv** – Original training dataset for 201 buildings.

- `building_id` – Unique building ID.  
- `timestamp` – When the measurement was taken.  
- `meter_reading` – Electricity consumption in kWh.  
- `anomaly` – Whether this reading is anomalous (1) or not (0).  

**building_meta.csv** – Metadata for 400+ buildings.

- `building_id` – Unique building ID.  
- `primary_use` – Primary use of the building (e.g., Office, Education).  
- `year_built` – Year the building was constructed.  
- `floor_count` – Number of floors.  

**train_meta_label.csv** – Processed training dataset ready for modeling.  

- Result of merging `train.csv` and `building_meta.csv` with **label encoding**.  
- Includes additional preprocessing steps such as:  
  - Handling missing values (`meter_reading_missing`, `year_built_missing`, `floor_count_missing`)  
  - Extraction of temporal features from `timestamp` (`hour`, `day`, `month`, `weekend_flag`)  

- Columns: `building_id`, `timestamp`, `meter_reading`, `square_feet`, `year_built`, `floor_count`, `primary_use_label`, `anomaly`, `meter_reading_missing`, `year_built_missing`, `floor_count_missing`  

**train_meta_one-hot.csv** – Same as above, but with **one-hot encoding** for `primary_use`. **Not yet used for training**.  

**merge.py** – Script used to merge `building_meta.csv` with `train.csv` to produce `train_meta_label.csv` and `train_meta_one-hot.csv`.  

**lead_train.ipynb** – Training and evaluation pipeline for anomaly detection.
- Input: `train_meta_label.csv`  
- Features: timestamp-derived features, metadata  
- Models: Logistic Regression, XGBoost, DNN  
- Outputs: trained models, evaluation metrics, loss/ROC plots

---

## 3. Data Preprocessing

The preprocessing steps applied to generate `train_meta_label.csv` include:

3.1. **Handling missing values**:
   - `meter_reading`: NaN → 0, add `meter_reading_missing` flag  
   - `year_built`: NaN → median(year_built), add `year_built_missing` flag  
   - `floor_count`: NaN → 1, add `floor_count_missing` flag  

3.2. **Feature extraction from timestamp**:
   - Derived features: `hour`, `day`, `month`, `weekend_flag` (1 if weekend)  

3.3. **Standardization**:
   - All numeric features scaled using `StandardScaler`  

3.4. **Handling class imbalance**:
   - Weighted loss applied during model training to improve learning on rare anomaly class  

---


## 4. Modeling
Three models were implemented:

### 4.1 Logistic Regression
- Iterations: 1000  
- Learning rate: 0.01  
- Optimization: Gradient Descent  

### 4.2 XGBoost (Best Model)
- Max depth: 3  
- Learning rate: 0.3  
- Number of estimators: 300  
- Scale_pos_weight: 20 (handle class imbalance)  
- Gamma: 0.2  
- Min_child_weight: 0  

### 4.3 Dense Neural Network (DNN)
- Architecture: 3 hidden layers  
- Optimizer: Adam (lr=0.0004)  
- Loss: Binary cross-entropy  
- Regularization: L2 + Dropout (0.5)  
- Early stopping: patience=10 epochs  
- Batch size: 64  

---

## 5. Evaluation & Results
Table 1: Comparison of AUC scores for different models on the validation set.

| Model             | AUC    |
|------------------|-------|
| Logistic Regression | 0.6076 |
| XGBoost           | 0.9921 |
| DNN               | 0.7135 |

Table 2: Precision, Recall, and F1-score for validation and test sets across models.
| Model               | Precision (Val) | Precision (Test) | Recall (Val) | Recall (Test) | F1-score (Val) | F1-score (Test) |
|--------------------|----------------|----------------|-------------|--------------|----------------|----------------|
| Logistic Regression | 0.56           | 0.56           | 0.58        | 0.58         | 0.57           | 0.57           |
| XGBoost             | **0.97**           | **0.97**           | **0.96**        | **0.95**         | **0.96**           | **0.96**           |
| DNN                 | 0.69           | 0.71           | 0.80        | 0.80         | 0.74           | 0.75           |


**Insights:**
- **XGBoost** shows the best performance with fast convergence and excellent balance between precision and recall for rare anomalies.  
- Logistic Regression performs poorly due to inability to capture non-linear relationships.  
- DNN performs reasonably but slightly lower than XGBoost.  

**Visualizations included:**
- Confusion matrices for the validation set (comparison of 3 models) (a. Logistic Regression, b. XGBoost, c. DNN)
<img width="900" height="306" alt="image" src="https://github.com/user-attachments/assets/4e39a0c8-df39-4c0b-8ba6-dfa325343731" />

- Confusion matrices for the test set (comparison of 3 models) (a. Logistic Regression, b. XGBoost, c. DNN)
<img width="900" height="309" alt="image" src="https://github.com/user-attachments/assets/5765e77d-70ef-4b92-aa87-863646b77557" />

- ROC curves
  <img width="914" height="656" alt="image" src="https://github.com/user-attachments/assets/8e52f04b-6d9a-4227-afc4-38cec311189c" />

---

## 6. Model Evaluation Summary

Logistic Regression showed the lowest performance among the three models. Its ability to handle non-linear or complex data is limited. Both precision and recall are balanced but low (<0.6) on the validation and test sets, indicating poor class separation. The ROC curve is close to random, with a low AUC (~0.6), showing performance slightly better than random guessing.

XGBoost is the most effective model, achieving high precision, recall, and F1-score. The AUC is ~0.99, indicating excellent classification performance.

DNN performs better than Logistic Regression but not as well as XGBoost. Its ROC curve is higher than Logistic Regression, demonstrating better class separation, but overall performance is moderate (AUC ~0.82).

Overall, considering the validation set results, XGBoost is the best-performing model, offering a good balance between precision and recall, suitable for tasks requiring accurate classification and anomaly detection.



