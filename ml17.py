
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score
)

# ==========================================
# PART 1: LINEAR REGRESSION
# ==========================================

# STEP 1: READ DATASET

file = input("Enter Admission_Predict.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# STEP 2: DATA PREPROCESSING

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# STEP 3: SELECT INPUT AND OUTPUT

X = df.drop(['Chance of Admit', 'Serial No.'], axis=1)
y = df['Chance of Admit']

# STEP 4: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# STEP 5: CREATE AND TRAIN LINEAR REGRESSION

lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict admission chances
y_pred = lr.predict(X_test)

# STEP 6: EVALUATE LINEAR REGRESSION

print("\nLINEAR REGRESSION RESULTS")

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# STEP 7: VISUALIZE LINEAR REGRESSION

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Admission Chance")
plt.ylabel("Predicted Admission Chance")
plt.title("Linear Regression - Actual vs Predicted")
plt.show()


# ==========================================
# PART 2: LOGISTIC REGRESSION
# ==========================================

# STEP 1: READ DATASET

file = input("Enter Social_Network_Ads.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# STEP 2: DATA PREPROCESSING

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove User ID
df = df.drop('User ID', axis=1, errors='ignore')

# Convert Gender into numbers
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# STEP 3: SELECT INPUT AND OUTPUT

X = df.drop('Purchased', axis=1)
y = df['Purchased']

# STEP 4: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# STEP 5: FEATURE SCALING

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# STEP 6: CREATE AND TRAIN LOGISTIC REGRESSION

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict purchase
y_pred = log_model.predict(X_test)

# STEP 7: EVALUATE LOGISTIC REGRESSION

print("\nLOGISTIC REGRESSION RESULTS")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

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

# STEP 8: VISUALIZE LOGISTIC REGRESSION

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

print("\nAssignment Completed Successfully!")
