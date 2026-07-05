#######################################

# Import necessary modules

#######################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

#######################################

# Define parameters and/or functions

#######################################

income = 80000.0
credit = 780
loan = 35180.59
years = 3

def calc_monthly_income(annual_income):
  return annual_income / 12

def month_loan_term(years):
  return years * 12

def calc_PMT(PV, t):
  i = 0.075 / 12 # For simiplicity of this simulation, all loans have a set annual interest rate of 7.5%
  PMT = PV * ((i * (1 + i)**t) / ((1 + i)**t - 1))
  return PMT

def calc_credit_score_threshold(monthly_income, PMT):
  percent_of_monthly_income = PMT / monthly_income

  if percent_of_monthly_income < 0.1:
    return 650
  elif percent_of_monthly_income < 0.2:
    return 740
  elif percent_of_monthly_income < 0.4:
    return 800
  else:
    return 851

def calc_loan_approval(annual_income, credit_score, loan_principal, years):
  monthly_income = calc_monthly_income(annual_income)
  t = month_loan_term(years)
  PMT = calc_PMT(loan_principal, t)
  credit_score_threshold = calc_credit_score_threshold(monthly_income, PMT)

  if credit_score >= credit_score_threshold:
    return "APPROVED"
  else:
    return "DENIED"
  
#######################################

# Load and preview data

#######################################

df = pd.read_csv("ML Example (Loan Approval).csv")
print(df.head(10))

#######################################

# Clean and preprocess data

#######################################

'''
This data set was generated in LibreOffice Calc (Excel) through the use of numerous intermediary calculation columns.
The logic that was built into those intermediary columns was recreated in the prior "Define parameters and/or functions" section. 
While the dataset does not accurately represent how real banks or other lending agencies assess and approve or deny loans, 
those details are considered out of scope for this excerise. I am primarily interested in creating a labeled dataset and 
training a logisitic regression model to see how accurately it can predict the likelihood that a given fictious loan would be 
approved or denied when going through this ficitious loan review process.
'''

# Remove intermediary calculation columns
out_of_scope_columns = ['income_lower', 'income_upper', 'months', 'PMT_calc_1', 'PMT_calc_2', 'PMT_calc_3', 'percent_of_monthly_income', 'below_10%_monthly_income', 'below_20%_monthly_income', 'below_40%_monthly_income', 'credit_score_threshold']
df.drop(columns=out_of_scope_columns, inplace=True)

print(df.head(10))

# While the dataset was created to be randomized, to further ensure random selection of training, test, and or validation datasets, scramble the orders of the rows in the dataframe.
df = df.sample(frac=1).reset_index(drop=True)
print(df.head(10))

# There are several columns whose currency values are formated as strings ($1,234.56). These must be converted to floats for proper processing.
currency_cols = ['Annual Income', 'Loan Principal ', 'Monthly Payment']

for col in currency_cols:
    df[col] = df[col].replace(r'[$,]', '', regex=True).astype(float)

print(df.head())
print(df.info())

#######################################

# Train model

#######################################

# Split the features and target to prevent data leakage. This prevents the model from seeing all the correct labels prematurely.
X = df.drop("Loan Approval", axis=1)
y = df["Loan Approval"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # Didn't realize that this module includes ways to randomize and stratify the dataset, but left the randomizer in the beginning just for future reference.

numeric_features = X.select_dtypes(include=['int64','float64']).columns
categorical_features = X.select_dtypes(include=['object','category']).columns

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

log_reg = LogisticRegression(max_iter=1000)

clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', log_reg)
])

clf.fit(X_train, y_train)

#######################################

# Test model

#######################################

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

#######################################

# Create entirely new test case to test model

#######################################

# Create a new ficitous loan application
new_applicant = pd.DataFrame([{
    'Annual Income': income,
    'Credit Score': credit,
    'Loan Principal ': loan,
    'Years': years,
    'Monthly Payment': calc_PMT(loan, month_loan_term(years))
}])

clf.predict(new_applicant)

probs = clf.predict_proba(new_applicant)[0]

loan_status = calc_loan_approval(income, credit, loan, years)

print(f"\n\nThe model predicts a {round(probs[0]*100,2)}% likelihood that the loan will be APPROVED.")
print(f"It predicts a {round(probs[1]*100,2)}% chance of denial.\n\n")

print(f"The original loan review logic would have {loan_status} this application.")