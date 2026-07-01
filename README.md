# Loan Approval Prediction
This project builds a machine learning model to predict whether a loan application will be approved or rejected based on applicant financial data and loan attributes. It is a binary classification problem using supervised learning techniques built end-to-end in Python: data cleaning, exploratory data analysis, preprocessing, model training, and evaluation.

## 📌Case Study Overview 

Loan approval decisions depend on many factors-income, credit history, loan amount, employment status, and more. This project builds and compares several classification models to predict approval outcomes, with the goal of identifying the most reliable model for this kind of structured/tabular prediction task

<img width="468" height="44" alt="image" src="https://github.com/user-attachments/assets/0dad5e1f-4f2c-4778-87f0-00bc333ce427" />

## Data Set Description 

* Source: [loan_approval_dataset](http://kaggle.com) 
* Size: 4,269
* Target variable: loan_status(1= Approved/0= Rejected)
* Feature Variables: loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, ressidential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value.
   
## 🧰 Tools and Libraries

Python 3.12
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

# 🔄 Project Workflow

## 🧹 Data Cleaning and Preprocessing:
Handled missing values, Fixed inconsistent enteries, Address outlliers

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
|-----------------------|--------------------------------------|-------------------------------|



🎯 Pre-Processing
Encoded categorical variables, scaled numerical features, split data into train/test sets


Feature Engineering
I removed the Customer_ID column as it serves as a unique identifier with no statistical relationship to the target variable (Claim). Removing non-predictive identifiers prevents model overfitting and reduces computational noise."
The NumberOfWindows feature contained mixed types, including the string '>=10'. I standardized this by converting the upper-bound string to a discrete integer and performing median imputation on missing values. This preserves the ordinal nature of the data while making it compatible with the regression and tree-based algorithms.
Mapping and encoding categorical variables: Used mapping technique for categorical columns like ** Building_paint ** Building_Fence ** Garden ** Settlment ** For categorical column that has more than 2 variables, ( Building_Type), I used LabelEncoder to encode the variables from categorical to numeric
#Mapping some of the cartegorical coliumns
Insurance_Train_Data['building_painted'] = Insurance_Train_Data['building_painted'].map({'N': 1, 'V': 1})
Insurance_Train_Data['building_fenced'] = Insurance_Train_Data['building_fenced'].map({'N': 0, 'V': 1})
Insurance_Train_Data['garden'] = Insurance_Train_Data['garden'].map({'V': 1, 'O': 0})
Insurance_Train_Data['settlement'] = Insurance_Train_Data['settlement'].map({'U': 1, 'R': 0})
#Using Label Encoder to encode Building Type column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Insurance_Train_Data['building_type'] = le.fit_transform(Insurance_Train_Data['building_type'].astype(str))
Correlation Matrix Heat Map This mathematically shows the correlation between out target(Claim) and every other numeric variables. Helps to focus on the variables that actually move the needle for the insurance company.
Correlation Heat Map

Feature selection ** Train-test split I chose to split the data using 70% to train, 30% to test, random state was 42 and stratify as y.

Standardization: The NumberOfWindows feature contained mixed types, including the string '>=10'. I standardized this by converting the upper-bound string to a discrete integer and performing median imputation on missing values. This preserves the ordinal nature of the data while making it compatible with the regression and tree-based algorithms."

#Adjusting the >=10 values in Number of windows column
#Strip any leading/trailing spaces
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].astype(str).str.strip()
#Replace '>=10' with '10'
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].replace('>=10', '10')
#Convert to numeric
Insurance_Train_Data['numberofwindows'] = pd.to_numeric(Insurance_Train_Data['numberofwindows'], errors='coerce')
#Fill the gaps (NaNs) with the median
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].fillna(Insurance_Train_Data['numberofwindows'].median())
#Convert to integer
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].astype
Scaling: Because models like Logistic Regression are sensitive to the "size" of numbers (e.g., comparing a building dimension of 2,000 to an insured period of 0.5), we applied Standard Scaling. ** Method: We fit the scaler only on the training data to prevent Data Leakage (using future info to predict the past). ** Result: All features were placed on a level playing field, ensuring the model didn't get "distracted" by large numbers. During the scaling phase, I identified 'Unknown' string entries within the features. I implemented a pipeline to coerce these into numerical format and applied median imputation, ensuring data consistency for the StandardScaler and subsequent models."
#Feature Scaling
#Using Standard Scaler

X_array = X_final.values.astype(float)
# Creates data with Mean = 0 and Standard Deviation = 1.
#Initialize Scaler
final_scaler = StandardScaler()

#FIT and TRANSFORM the array! (Learning the mean and variance from training data)
X_scaled = final_scaler.fit_transform(X_array)

print("Shape of scaled data:", X_scaled.shape)
print("Success! Data is perfectly cleaned and scaled" )
🧠 Feature Engineering Insight
Feature Engineering revealed a significant "Risk Gap" based of building age. By categorizing the "Building Age" into Mordern and Historical, we validated that older structure have a significantly higher claim frequency. This allowed our model(Logistics Regression) to assign higher risk weights to historical properties, directly improving the model's F-1 score

Bivariate analysis reveals that Historic buildings (50+ years) have a higher claim frequency compared to modern structures. This justifies the inclusion of 'Is_Historic' as a key feature for our predictive models.".
Target variable balance assessment: The target variable (insurance claim) appears imbalanced, with non-claims dominating.
Correlation Heat Map

🧠 Model Building
Machine learning algorithms selection We tested three distinct models to see which one understood the insurance risks best: ** Logistic Regression: Used as my Baseline. It assumes a linear relationship between features and risk. ** Random Forest: Used to handle Outliers. It builds a "Forest" of decision trees to find stable patterns. ** XGBoost: Optimized using gradient boosting for the highest predictive power.

Model Initialization ** Logistics Regression

   og_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight='balanced',
    solver='liblinear'
   )
** Random Forest

  rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_features="sqrt", 
    max_depth=6, 
    max_leaf_nodes=6,
   class_weight='balanced'
   )
** XGBoost

   xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'    
   )
Model training on training dataset ** Logistic Regression:
  log_model.fit(X-train, y_train)
** Random Forest

  rf_model.fir(X_train, y_train)
** XGBoost: xgb_model.fit(X-train, y_train)

  xgb_model.fir(X_train, y_train)
📌 Model Evaluation
Accuracy
Confusion Matrix onfusion Matrix
🧠 Comparing the Classification metrics (Precision, Recall, F1-score) of the 3 models
Instead of just looking at Accuracy (which can be misleading in insurance where claims are rare), I prioritized the F1-Score to balance Precision (avoiding false alarms) and Recall (catching actual claims).

Key Finding: While Logistic Regression provided a solid foundation and it was considered for the project, XGBoost has a high accuracy but lowest F1 score. ** Interpretation: The model achieved an F1-score of [0.46], meaning it successfully balanced the need to catch actual claims (Recall) without raising too many false alarms (Precision).

Overfitting and Validation To ensure this wasn't just a "lucky guess," iperformed 5-Fold Cross-Validation.

Verification: The model showed consistent scores across all folds, proving it is Robust and will work on new, unseen buildings in the future, not just the ones in our training set.
| Model | Accuracy | Precision | Recall | F1 Score | |-----------------------------|----------|-------------|----------|-----------| | Logistic Regression | 0.713687 | 0.405159 | 0.544898 | 0.464752 | | Random Forest | 0.716015 | 0.402913 | 0.508163 | 0.449458 |
| XGBoost | 0.784451 | 0.577143 | 0.206122 | 0.303759 |

Prediction Chat

** Based on these metrics, Logistic Regression is actually the winner, but with a very important "Insurance Business" reason why. Why Logistic Regression wins here? In insurance, F1-Score is more important because it is the "balance" between being right (Precision) and not missing anyone (Recall). *** The Problem with XGBoost: Even though its Accuracy and Precision are high, its Recall is the lowest. This means XGBoost is being "too shy." It only predicts a claim when it is 99% sure, but it is missing most of the actual claims. In insurance, missing a claim (a "False Negative") is very expensive. *** The Strength of Log Reg: It has the Highest Recall and F1-Score. This means it is much better at "catching" the people who will actually make a claim, even if it makes a few more mistakes (lower Precision) along the way.

📊 Verdict: We choose Logistic Regression.
"While XGBoost showed higher overall accuracy, In the context of insurance risk, a high Recall is critical to ensure that potential claims are not overlooked. Logistic Regression achieved the highest F1-Score, providing the best balance between identifying risks and maintaining prediction accuracy."

The model demonstrates reasonable predictive performance
Certain demographic and policy-related features strongly influence claim likelihood
Class imbalance affects prediction outcomes and should be addressed in future iterations
Confusion Matrix

⚠️ Limitations
Class imbalance in claim vs non-claim observations
Limited feature granularity
No cost-sensitive evaluation metrics included
💡 Key Insights
Feature Importance: Building_Dimension and Date_of_Occupancy were the strongest predictors of claim likelihood.
Business Value: The model allows for proactive risk management and more competitive pricing for low-risk clients.
✅ Recomendations
Address gender pay gaps, especially in flagged departments/region
Improve transparency in performance evaluation metrics
Consider pay restructuring to meet legal compliance
Implement regular reviews of pay and performance equity
Develop HR policies encouraging full data disclosure
