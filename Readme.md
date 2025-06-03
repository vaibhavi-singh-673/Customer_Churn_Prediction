# Customer Churn Prediction

## Project Overview

This project aims to develop a machine learning pipeline for predicting customer churn in a telecommunications company. Churn is defined as customers unsubscribing or cancelling their service. By analyzing historical customer data, this project seeks to identify patterns and features that indicate a higher risk of churn, enabling the business to take proactive steps to retain at‐risk customers.

**Key objectives:**

- Perform exploratory data analysis (EDA) to understand the distribution and relationships of features.
- Preprocess and clean the data (handle missing values, encode categorical variables, scale numerical features).
- Perform feature selection to remove redundant or irrelevant features.
- Train and evaluate multiple classification models (Random Forest, Logistic Regression, Decision Tree, K-NN, Gaussian Naive Bayes).
- Compare model performances and identify the best approach for churn prediction.

---

## Dataset

The dataset used in this project is the **Telco Customer Churn Dataset**, which was originally sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). A copy of the dataset (CSV file) is included in the project’s `data/` directory.

### Data Directory Structure

```

├── data/
│ ├── customer_churn.csv
├── customer_churn_prediction.ipynb
├── README.md

```

> **Note:**
>
> - `customer_churn.csv` is the raw Telco Customer Churn data.
> - `customer_churn_prediction.ipynb` contains the entire analysis, from EDA through modeling.

### Feature Descriptions

Each row in the dataset represents a single customer. Below is a subset of the most important columns:

| Column Name        | Description                                                                          |
| ------------------ | ------------------------------------------------------------------------------------ |
| `customerID`       | Unique identifier for each customer                                                  |
| `gender`           | Whether the customer is male or female                                               |
| `SeniorCitizen`    | Whether the customer is a senior citizen (1 = Yes, 0 = No)                           |
| `Partner`          | Whether the customer has a partner (Yes/No)                                          |
| `Dependents`       | Whether the customer has dependents (Yes/No)                                         |
| `tenure`           | Number of months the customer has stayed with the company                            |
| `PhoneService`     | Whether the customer has phone service (Yes/No)                                      |
| `MultipleLines`    | Whether the customer has multiple lines (Yes/No/No phone service)                    |
| `InternetService`  | Customer’s internet provider (DSL/Fiber optic/No)                                    |
| `OnlineSecurity`   | Whether the customer has online security add-on (Yes/No/No internet service)         |
| `OnlineBackup`     | Whether the customer has online backup add-on (Yes/No/No internet service)           |
| `DeviceProtection` | Whether the customer has device protection add-on (Yes/No/No internet service)       |
| `TechSupport`      | Whether the customer has tech support add-on (Yes/No/No internet service)            |
| `Contract`         | Contract term of the customer (Month-to-month/One year/Two year)                     |
| `PaperlessBilling` | Whether the customer has paperless billing (Yes/No)                                  |
| `PaymentMethod`    | Payment method (Electronic check/Mailed check/Bank transfer credit card/Credit card) |
| `MonthlyCharges`   | The amount charged to the customer monthly                                           |
| `TotalCharges`     | The total amount charged to the customer                                             |
| `Churn`            | Whether the customer churned (Yes = churned, No = stayed)                            |

> **Important:**
>
> - `"Churn"` is the target variable.
> - Numerical variables (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`) will require scaling.
> - Many categorical features must be encoded before modeling.

---

## Environment Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vaibhavi-singh-673/Customer_Churn_Prediction.git
   cd Customer_Churn_Prediction
   ```

2. **Create a Virtual Environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   # venv\Scripts\activate        # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **requirements.txt** (example)
   >
   > ```txt
   > pandas>=1.2.0
   > numpy>=1.19.0
   > matplotlib>=3.3.0
   > seaborn>=0.11.0
   > scikit-learn>=0.24.0
   > jupyterlab>=3.0.0
   > ```

4. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook data/customer_churn_prediction.ipynb
   ```

   - Open `customer_churn_prediction.ipynb` and run all cells to reproduce the analysis.

---

## Project Structure

```plaintext
├── data/
│   └── customer_churn.csv
│
├── customer_churn_prediction.ipynb
├── requirements.txt
└── README.md
```

- **data/**

  - `customer_churn.csv`: Raw dataset downloaded from Kaggle.
  - `customer_churn_prediction.ipynb`: Jupyter notebook containing full EDA, preprocessing, modeling, and evaluation.

- **requirements.txt**
  Contains all Python packages needed to run the notebook (pandas, numpy, matplotlib, seaborn, scikit-learn, etc.).

- **README.md**
  This file, providing an overview of the project, instructions, and notes.

---

## Detailed Workflow

### 1. Exploratory Data Analysis (EDA)

- **Load Data**

  ```python
  import pandas as pd
  df = pd.read_csv('data/customer_churn.csv')
  ```

- **Inspect Basic Information**

  - `df.shape` → shows total rows and columns.
  - `df.head()` → displays first few rows.
  - `df.info()` → reveals data types and null values.
  - `df.describe()` → summary statistics for numerical features.

- **Target Variable Distribution**

  - Check churn count:

    ```python
    df['Churn'].value_counts()
    ```

  - Compute churn rate (percentage of customers who churned).

- **Missing / Erroneous Values**

  - Check for blank strings or spaces in `TotalCharges` (which may be stored as object).
  - Convert `TotalCharges` to numeric and handle missing/invalid entries (e.g., set to NaN or drop those rows).

- **Univariate Analysis**

  - Plot distributions of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using histograms or boxplots.
  - Visualize counts of categorical features (`Contract`, `PaymentMethod`, `InternetService`, etc.).

- **Bivariate Analysis**

  - Correlation heatmap for numerical variables.
  - Bar plots/boxplots of churn vs. key numerical features (e.g., monthly charges vs. churn status).
  - Count plots of churn vs. categorical features.

### 2. Data Preprocessing

1. **Handle Missing Values**

   - Drop rows where `TotalCharges` is blank or cannot be converted to numeric.
   - Verify no other missing values remain.

2. **Convert Data Types**

   - Convert `TotalCharges` to `float64`.
   - Convert `SeniorCitizen` from numeric (0/1) to categorical (`No`/`Yes`) if desired, or keep numeric.

3. **Encode Categorical Variables**

   - **Binary Features (Yes/No):** map to 0/1:

     ```python
     binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
     for col in binary_cols:
         df[col] = df[col].map({'Yes': 1, 'No': 0})
     ```

   - **Multiple Categories:** apply one-hot encoding (dummy variables) for features like `InternetService`, `Contract`, `PaymentMethod`, etc.

4. **Feature Engineering**

   - Drop `customerID` (unique identifier, not predictive).
   - Create new features if applicable (e.g., grouping tenure into buckets).
   - Scale numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` or `MinMaxScaler`.

5. **Train/Test Split**

   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop('Churn', axis=1)
   y = df['Churn']
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

6. **Feature Scaling**

   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

### 3. Feature Selection

- **Random Forest Feature Importance**

  1. Train a Random Forest classifier on the scaled training data.
  2. Extract `.feature_importances_` and rank features by importance.
  3. Plot a bar chart of feature importances to visually inspect which features contribute most.

- **Recursive Feature Elimination (RFE) with Logistic Regression**

  1. Use `RFECV` from `sklearn.feature_selection` with Logistic Regression (L1 penalty) to identify the optimal subset of features.
  2. Observe the number of selected features (e.g., 18).
  3. If the RFE suggests all features are useful, retain the full set; otherwise, drop low-importance features.

### 4. Model Building & Evaluation

The following models are trained and evaluated:

1. **Random Forest Classifier**

   - Hyperparameters: `n_estimators=100`, `random_state=0`.
   - Fit on `X_train_scaled`, evaluate on `X_test_scaled`.

2. **Logistic Regression**

   - Hyperparameters: `penalty='l1'`, `C=0.1`, `solver='liblinear'` (if using L1).
   - Fit on scaled data, evaluate performance.

3. **Decision Tree Classifier**

   - Default settings or tuned criteria (e.g., `max_depth`, `min_samples_split`).

4. **K-Nearest Neighbors (K-NN)**

   - Hyperparameters: `n_neighbors=5` (default).
   - Evaluate on scaled features (K-NN is sensitive to feature scale).

5. **Gaussian Naive Bayes**

   - Fit on the training set, evaluate on the test set.

For each model:

- Fit on the training set.
- Generate predictions on the test set.
- Compute evaluation metrics:

  - **Accuracy**: overall correctness.
  - **Precision, Recall, F1-score**: via `classification_report`.
  - **ROC-AUC score** (optional, but recommended for imbalanced data).
  - **Confusion matrix**: visualize true positives, false positives, etc.

Example code snippet for evaluation:

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Train model (e.g., Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"Random Forest ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()
```

---

## Results Summary

> **Note:** Detailed model results (accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices) are displayed in the Jupyter notebook. Below is a general summary outline—please refer to the notebook for exact numbers.

- **Random Forest**

  - Achieved high accuracy and balanced precision/recall.
  - ROC-AUC typically above 0.80.
  - Feature importance indicates that `MonthlyCharges` and `tenure` are among the most predictive variables.

- **Logistic Regression (L1)**

  - Competitive accuracy, with the added benefit of feature sparsity (identifies and drops irrelevant features).
  - Good interpretability when examining coefficients.

- **Decision Tree**

  - Fast training, but may overfit if not pruned or depth-limited.
  - Performance slightly lower than Random Forest in many cases.

- **K-Nearest Neighbors (K-NN)**

  - Sensitive to feature scaling; performance often moderate.
  - Requires careful tuning of `k`.

- **Gaussian Naive Bayes**

  - Very fast to train and predict.
  - Assumes feature independence; performance can be lower if features are correlated.

The **best-performing** model on this dataset was typically the **Random Forest Classifier**, balancing accuracy and generalization.

---

## How to Reproduce / Usage

1. **Ensure dataset is in `data/customer_churn.csv`.**

   - If you obtained the dataset directly from Kaggle, rename it `customer_churn.csv` and place it under `data/`.

2. **Activate the virtual environment** (if not already active):

   ```bash
   source venv/bin/activate    # macOS/Linux
   # venv\Scripts\activate     # Windows
   ```

3. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook data/customer_churn_prediction.ipynb
   ```

   - Run each cell sequentially to reproduce the analysis, plots, and model training.

4. **Inspect Model Results**:

   - Scroll to the “Model Building & Evaluation” section in the notebook to see accuracy scores, classification reports, ROC curves, and confusion matrices for each model.

---

## Future Work

1. **Hyperparameter Tuning**

   - Perform grid search or randomized search (e.g., using `GridSearchCV` or `RandomizedSearchCV`) for each model to optimize hyperparameters.

2. **Address Class Imbalance**

   - Although the churn rate is not extremely imbalanced, consider techniques such as SMOTE, ADASYN, or class-weighting to further improve recall for the churn class.

3. **Time-based Features**

   - Incorporate additional temporal features, such as monthly payment trends or changes in service usage over time.

4. **Advanced Algorithms**

   - Experiment with gradient boosting models (XGBoost, LightGBM, CatBoost) to potentially achieve higher performance.

5. **Deployment**

   - Wrap the best model into a Flask/FastAPI web service or create a simple API endpoint.
   - Build a dashboard (e.g., with Dash or Streamlit) for real-time churn probability predictions based on user inputs.

---

## Acknowledgements

- **Dataset Source**: “Telco Customer Churn” dataset from Kaggle ([https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)).
- **Libraries & Tools**:

  - [pandas](https://pandas.pydata.org/) for data manipulation
  - [NumPy](https://numpy.org/) for numerical operations
  - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for visualization
  - [scikit-learn](https://scikit-learn.org/) for modeling, preprocessing, and evaluation
  - [Jupyter Notebook](https://jupyter.org/) for interactive analysis

---

## References

1. _Telco Customer Churn Dataset_. Kaggle.
   URL: [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

2. Géron, Aurélien. _Hands-On Machine Learning with Scikit-Learn & TensorFlow_. O’Reilly Media, 2017.
   (For general EDA and modeling guidance.)

3. Scikit-learn documentation:

   - Preprocessing: [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
   - Model Selection & Evaluation: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)

---

**Made by Vaibhavi Singh** – https://github.com/vaibhavi-singh-673
