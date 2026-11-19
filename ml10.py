
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score
)

# ==========================================
# STEP 1: READ DATASET
# ==========================================

file = input("Enter madfhantr.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# STEP 3: CONVERT CATEGORICAL DATA INTO NUMBERS
# ==========================================

df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

X = df.drop(['Loan_Status', 'Loan_ID'], axis=1, errors='ignore')
X = pd.get_dummies(X, drop_first=True)

y = df['Loan_Status']

print("\nLoan Status Classes:")
print(y.value_counts())

# ==========================================
# STEP 4: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================================
# STEP 5: CREATE AND TRAIN DECISION TREE
# ==========================================

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict loan status
y_pred = dt.predict(X_test)

# ==========================================
# STEP 6: EVALUATE MODEL
# ==========================================

print("\nDECISION TREE RESULTS")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred, zero_division=0))

print("\nRecall:")
print(recall_score(y_test, y_pred, zero_division=0))

print("\nF1 Score:")
print(f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# STEP 7: VISUALIZE DECISION TREE
# ==========================================

plt.figure(figsize=(18, 10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Rejected', 'Approved'],
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree - Loan Eligibility Prediction")
plt.show()

print("\nAssignment Completed Successfully!")
