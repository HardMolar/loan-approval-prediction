# Loan Approval Prediction
This project builds a machine learning model to predict whether a loan application will be approved or rejected based on applicant financial data and loan attributes. It is a binary classification problem using supervised learning techniques built end-to-end in Python: data cleaning, exploratory data analysis, preprocessing, model training, and evaluation.

 ## 📌Case Study Overview 

Loan approval decisions depend on many factors-income, credit history, loan amount, employment status, and more. This project builds and compares several classification models to predict approval outcomes, with the goal of identifying the most reliable model for this kind of structured/tabular prediction task

## 📊 Data Set Description 

* Source: [loan_approval_dataset](http://kaggle.com) 
* Size: 4,269
* Target variable: loan_status(1= Approved/0= Rejected)
* Feature Variables: loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, ressidential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value.
   
## 🧰 Tools and Libraries

* Python 3.12
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

# 🔄 Project Workflow

## 🧹 Data Cleaning and Preprocessing:
* Handled missing values
* Fixed inconsistent enteries
* Address outlliers

## 🚧 Exploratory Data Analysis (EDA)

The following EDA steps were performed to understand the dataset before modeling:

* Loan Status Distribution: Checked class balance between approved and rejected loans
* Numerical Feature Visualization: Histograms to understand distributions of income, loan amount, and assets
* Outlier Detection: Boxplots on numerical columns to identify extreme values
* Outcome Proportionality: Pie chart showing approval vs rejection percentages
* Missing Values Analysis: Heatmap and count check — no missing values found
* Categorical vs Loan Status: Countplots for education and self_employed against loan_status
* Numerical vs Loan Status: Boxplots comparing feature distributions across approved and rejected
* Correlation Heatmap: Heatmap of all numerical feature correlations

 ### 🚧 Key EDA Findings
 #### cibil_score
 * Findings: Median ~700 for approved vs ~420 for rejected. Strongest separator
 * Decision: Keep — priority feature
 #### income_annum
 * Findings: Almost identical distribution for approved and rejected
 * Decision: Keep — weak alone, useful in ratios
 #### loan_amount
 * Findings: Similar distributions across both classes
 * Decision: Keep — use in ratio features
 #### All asset features
 * Findings: Near identical distributions for both classes
 * Decision: Keep — combine into total_assets
 #### education
 * Findings: No meaningful difference in approval rates
 * Decision: Drop — weak predictor
 #### self_employed
 * Findings: No meaningful difference in approval rates
 * Decision: Drop — weak predictor
 #### no_of_dependants
 * Findings: Near zero correlation with all features
 * Decision: Drop — weak predictor
 #### loan_term
 * Findings: Near zero correlation with all features
 * Decision: Drop — weak predictor
 #### income_annum vs loan_amount
 * Findings: Correlation of 0.93 — multicollinearity risk
 * Decision: Engineer ratio instead
 #### income_annum vs luxury_assets
 * Findings: Correlation of 0.93 — highly redundant
 * Decision: Combine into engineered features

## 📊 Data Pre-Processing

### Data Cleaning
* Reloaded dataset fresh from CSV to avoid corrupted state
* Stripped leading and trailing spaces from all column names using .str.strip()
* Stripped spaces from all string/object column values to fix encoding issues
* Encoded target variable: loan_status mapped to Approved=1, Rejected=0

### Encoding
Three new features were engineered from existing ones to improve model signal:

#### income_to_loan_ratio
* Formular: income_annum/loan_amount
* Purpose: Can the applicant afford the loan?
#### total_assets
* Formular: Sum of all 4 asset columns
* Purpose: Overall financial strength
#### assets_to_loan_ratio
* Formular: total_assets / loan_amount
* Purpose: Is the applicant sufficiently backed?

### Dropped Features
The following features were dropped based on EDA insights:
* education — no meaningful difference in approval rates between Graduate and Not Graduate
* self_employed — approval rates identical for both employed and self-employed
* no_of_dependents — near zero correlation with loan_status
* loan_term — near zero correlation with loan_status

### Train/Test Split and Scaling
* Train/Test Split: 70% training, 30% test (random_state=42)
* Scaling: StandardScaler applied — fit on training data only, transform on both sets
* Scaler saved as scaler.pkl alongside the model to ensure consistent scaling at inference

## 🧠 Model Training
Three models were trained and compared to identify the best performer:
### Logistic Regression
* Purpose: Baseline linear model — simple and interpretable
### Random Forest
* Purpose: Ensemble of decision trees — handles non-linearity well
### XGBoost
* Purpose: Gradient boosting — high performance on tabular data

## 📌 Model Evaluation Result
### Accuracy Comparison

| Model              | Accuracy         |F1-Score(Weighted) | Verdict                   |
|--------------------|------------------|-------------------|---------------------------|
|Logistics Regression| 91.18%           | 0.91              | Goood Baseline            |
|Radom Forest        | 95.24%           | 0.95              | Best Overall              |
|XGBoost             | 94.54%           | 0.95              | Strong runner-up          |


### Confusion Matrix Analysis

| Model              |True Negative(TN) |True Positive(TP)  |False Positive(FP) |False Negative(FN) |
|--------------------|------------------|-------------------|------------------ |-------------------|
|Logistics Regression| 425              | 743               | 46                | 67                |
|Radom Forest        | 469              | 751               | 2                 | 59                |
|XGBoost             | 455              | 756               | 16                | 54                |

In a real banking context, False Positives (approving loans that should be rejected) are the most costly error. Random Forest virtually eliminates this with only 2 false approvals(Positives) — making it the safest and most reliable model

## 📌 Final Model Selection
Selected Model: Random Forest Classifier
| Metric                     | Value                           |
|----------------------------|---------------------------------|
|Accuracy                    | 95.24%                          |
|Precision(Approved)         | 1.00                            | 
|Recall(Approved)            | 0.93                            |
|F1-Score                    | 0.95                            |
|False Approvals             | Only 2 out of 471 rejected cases|
|Saved As                    | best_model_random_forest.pkl    |

## ✅ Model Saving
Both the trained model and the scaler were saved using joblib for future use:
* best_model_random_forest.pkl — the trained Random Forest model
* scaler.pkl — the StandardScaler fitted on training data

## 💡 Conclusion 
This project successfully built a loan approval prediction system with 95.24% accuracy using Random Forest. Key takeaways include:

* cibil_score is by far the most powerful predictor of loan approval in this dataset
* Income and asset features individually show weak predictive power but become useful through feature engineering
* Random Forest outperformed both Logistic Regression and XGBoost, especially in minimizing false approvals
* Feature engineering (income_to_loan_ratio, total_assets, assets_to_loan_ratio) added meaningful signal beyond raw features
* Proper data preprocessing — especially stripping spaces and correct encoding — was critical to avoid pipeline errors
* The model is saved and ready for deployment into a real application or API


Implement regular reviews of pay and performance equity
Develop HR policies encouraging full data discl
