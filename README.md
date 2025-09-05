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

**train_meta_one-hot.csv** – Same as above, but with **one-hot encoding** for `primary_use`. **Not yet used for training**.  

**merge.py** – Script used to merge `building_meta.csv` with `train.csv` to produce `train_meta_label.csv` and `train_meta_one-hot.csv`.  

**lead_train.ipynb** – Training and evaluation pipeline for anomaly detection.
- Input: `train_meta_label.csv`  
- Outputs: trained models, evaluation metrics, loss/ROC plots

---

## 3. Data Preprocessing

The preprocessing steps applied to generate `train_meta_label.csv` include:

**Handling missing values**:
   - `meter_reading`: NaN → 0, add `meter_reading_missing` flag  
   - `year_built`: NaN → median(year_built), add `year_built_missing` flag  
   - `floor_count`: NaN → 1, add `floor_count_missing` flag  

**Feature extraction from timestamp**- Derived features: `hour`, `day`, `month`, `weekend_flag` (1 if weekend)
     
**Standardization** - All numeric features scaled using `StandardScaler`  

**Handling class imbalance** - Weighted loss applied during model training to improve learning on rare anomaly class  

---


## 4. Modeling
The dataset is split into three subsets: training set (70%), validation set (27%) and test set (3%), ensuring that the ratio of anomalous vs. normal samples is preserved in each split. All numerical features are standardized using StandardScaler to improve model convergence and stability.

I implemented three complementary models to detect anomalies in building energy consumption:

4.1. Logistic Regression
### 4.1 Logistic Regression

Key characteristics:
- Outputs a probability between 0 and 1, which can be thresholded to assign class labels (0 or 1).  
- Provides **interpretable coefficients**, allowing insight into how each feature contributes to the prediction.  
- Serves as a strong **baseline model** for anomaly detection, especially when combined with feature engineering.  

**Advantages in this project:**
- Fast to train and easy to interpret.  
- Provides a transparent benchmark for evaluating more complex models (XGBoost, DNN).  
- Works effectively when anomalies are rare and features are properly scaled.  

> This model serves as a baseline for anomaly detection, leveraging the sigmoid function to map feature combinations into probabilistic predictions.


4.2. XGBoost Classifier

Key characteristics:
- Builds an ensemble of **decision trees**, where each new tree corrects the errors of previous ones.
- Optimizes a **regularized objective function** to prevent overfitting.
- Handles **class imbalance** via parameters like `scale_pos_weight`.

**Advantages in this project:**
- High accuracy for detecting anomalies in energy consumption data.
- Robust to outliers and irrelevant features.
- Capable of capturing complex, non-linear relationships between building metadata and electricity usage patterns.

> XGBoost serves as the **best-performing model** in this project, effectively detecting anomalies in building energy consumption by leveraging an ensemble of boosted decision trees.

4.3. Dense Neural Network (DNN)

Multi-layer feedforward network capturing complex feature interactions.

Architecture: 5 hidden layers with Dropout and L2 regularization.

Early stopping: 10 epochs patience

<img width="916" height="184" alt="image" src="https://github.com/user-attachments/assets/e2b76fac-adfc-47a1-966f-887f503ce5f4" />

**Figure 4.1: Network architecture used in the DNN model for anomaly detection**


## 5. Evaluation & Results
Table 5.1: Comparison of AUC scores for different models on the validation set.

| Model             | AUC    |
|------------------|-------|
| Logistic Regression | 0.6076 |
| XGBoost           | 0.9921 |
| DNN               | 0.7135 |

Table 5.2: Precision, Recall, and F1-score for validation and test sets across models.
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

**Figure 5.1**: Confusion matrices for the validation set (comparison of 3 models) (a. Logistic Regression, b. XGBoost, c. DNN)

<img width="900" height="306" alt="image" src="https://github.com/user-attachments/assets/4e39a0c8-df39-4c0b-8ba6-dfa325343731" />

**Figure 5.2**: Confusion matrices for the test set (comparison of 3 models) (a. Logistic Regression, b. XGBoost, c. DNN)

<img width="900" height="309" alt="image" src="https://github.com/user-attachments/assets/5765e77d-70ef-4b92-aa87-863646b77557" />

**Figure 5.3**: ROC curves

  <img width="914" height="656" alt="image" src="https://github.com/user-attachments/assets/8e52f04b-6d9a-4227-afc4-38cec311189c" />

---

## 6. Model Evaluation Summary

Logistic Regression showed the lowest performance among the three models. Its ability to handle non-linear or complex data is limited. Both precision and recall are balanced but low (<0.6) on the validation and test sets, indicating poor class separation. The ROC curve is close to random, with a low AUC (~0.6), showing performance slightly better than random guessing.

XGBoost is the most effective model, achieving high precision, recall, and F1-score. The AUC is ~0.99, indicating excellent classification performance.

DNN performs better than Logistic Regression but not as well as XGBoost. Its ROC curve is higher than Logistic Regression, demonstrating better class separation, but overall performance is moderate (AUC ~0.82).

Overall, considering the validation set results, XGBoost is the best-performing model, offering a good balance between precision and recall, suitable for tasks requiring accurate classification and anomaly detection.



