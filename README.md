# Online-Payment-Fraud-detection

## Project Overview
This project focuses on detecting fraudulent online transactions for Blossom Bank. By analyzing transaction data and implementing machine learning algorithms, the system aims to accurately classify transactions as either fraudulent or legitimate. This is crucial for improving fraud detection mechanisms and ensuring customer safety.

---

## Key Features
- **Exploratory Data Analysis (EDA)**:
  - Univariate, bivariate, and multivariate analysis to uncover patterns in transaction data.
  - Visualizations to understand transaction types, amounts, and fraudulent activities.
- **Data Preprocessing**:
  - Handled missing values and performed feature engineering, including encoding categorical variables.
- **Machine Learning Models**:
  - Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors classifiers were implemented and evaluated.
- **Model Validation**:
  - Cross-validation was conducted to ensure model robustness and reliability.
- **Fraud Analysis**:
  - Focused on minimizing false negatives to prioritize detecting fraudulent transactions.

---

## Dataset Details
- **Source**: The dataset for this project is available on Kaggle. Due to its size (over 6 million rows and 100MB), it is not included in this repository.
  - **Download the dataset here**: [Online Payment Fraud Detection Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data)
- **Dataset Details**:
  - **Rows**: 6,362,620
  - **Columns**: 10, including:
    - `step`: Time step of the transaction.
    - `type`: Type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT).
    - `amount`: Amount involved in the transaction.
    - `fraud_transaction`: Label indicating whether the transaction is fraudulent (1) or not (0).
- **Balance**: The dataset is imbalanced, with only ~13% of transactions marked as fraudulent.

---

## Methodology
### 1. **Data Preprocessing**
   - Dropped unnecessary columns.
   - Encoded categorical variables like `type` using one-hot encoding.
   - Standardized feature scaling for model input.

### 2. **Exploratory Data Analysis**
   - Analyzed transaction types and identified that "CASH_OUT" and "TRANSFER" transactions are most prone to fraud.
   - Visualized the distribution of transaction amounts and their relation to fraudulent behavior.

### 3. **Feature Engineering**
   - Generated dummy variables for categorical features.
   - Dropped irrelevant columns such as customer identifiers for better model performance.

### 4. **Model Selection**
   - Implemented and compared the following models:
     - Logistic Regression
     - K-Nearest Neighbors
     - Decision Tree Classifier
     - Random Forest Classifier
   - Evaluated models using accuracy, precision, recall, and F1-score.

### 5. **Model Validation**
   - Performed cross-validation with a focus on recall to prioritize minimizing false negatives.

---

## Results
- **Best Model**: Decision Tree Classifier
  - Accuracy: 99.97%
  - Recall (cross-validation): 91%
  - High recall ensures fraudulent transactions are correctly identified.
- **Random Forest Classifier** also performed well with slightly lower recall but higher overall accuracy.

---

## Visual Insights
1. **Transaction Types**:
   - "CASH_OUT" and "PAYMENT" are the most common types.
   - "TRANSFER" transactions involve the highest amounts.
2. **Fraud Distribution**:
   - Most fraudulent transactions involve amounts below $10,000,000.
3. **Fraud by Transaction Type**:
   - Majority of fraudulent transactions are observed in "CASH_OUT" and "TRANSFER."

---

## Requirements
- **Programming Languages**: Python
- **Libraries**:
  - pandas, numpy: Data manipulation
  - matplotlib, seaborn: Visualization
  - scikit-learn: Machine learning
- **Environment**: Jupyter Notebook

---

## Installation and Usage
1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/online-payment-fraud-detection.git
   cd online-payment-fraud-detection
   Download the Dataset:

2. Download the dataset from Kaggle: Online Payment Fraud Detection Dataset
Save the file in the project directory as fraud.csv.
3. Install Dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
5. Run the Notebook: Open and execute ipynb file in Jupyter Notebook.

## Future Scope
Real-Time Detection: Integrate the model into a real-time transaction monitoring system.
Advanced Techniques: Experiment with deep learning models like neural networks for improved accuracy.
Imbalance Handling: Use SMOTE or similar techniques to handle dataset imbalance more effectively.
Feature Expansion: Include additional features such as geographic location and time zones to improve predictions.

Thank you for exploring this project! Your feedback is greatly appreciated.
