
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

file = input("Enter pima-indians-diabetes.csv file path: ").strip().strip('"')
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

# Replace invalid zero values with NaN
cols = ['Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI']

df[cols] = df[cols].replace(0, pd.NA)

# Fill missing values using median
for col in cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# STEP 3: SELECT INPUT AND OUTPUT
# ==========================================

X = df.drop('Outcome', axis=1)
y = df['Outcome']

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

# Predict diabetes outcome
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
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree - Diabetes Prediction")
plt.show()

print("\nAssignment Completed Successfully!")
