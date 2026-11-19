
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Step 1: Load dataset
file = input("Enter CSV file path: ").strip().strip('"')
df = pd.read_csv(file)

df.columns = df.columns.str.strip()

print("\nFirst 5 Records:")
print(df.head())

# Step 2: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Remove duplicate records
df = df.drop_duplicates()

# Step 4: Handle missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 5: Drop ID columns
for col in ['ID', 'ZIP Code']:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Step 6: Label Encoding
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Step 7: Initialize independent and dependent variables
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']

# Step 8: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 9: Feature Scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 10: Create SVM model
svm = SVC(kernel='linear', random_state=42)

# Step 11: Train the model
svm.fit(X_train, y_train)

print("\nSVM Model Trained Successfully!")

# Step 12: Predict values
y_pred = svm.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# Step 13: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 14: Accuracy
print("\nAccuracy:")
print(metrics.accuracy_score(y_test, y_pred))

# Step 15: Precision
print("\nPrecision:")
print(metrics.precision_score(y_test, y_pred, zero_division=0))

# Step 16: Recall
print("\nRecall:")
print(metrics.recall_score(y_test, y_pred, zero_division=0))

# Step 17: F1 Score
print("\nF1 Score:")
print(metrics.f1_score(y_test, y_pred, zero_division=0))

# Step 18: Confusion Matrix Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("SVM Confusion Matrix")
plt.show()

# Step 19: Conclusion
print("\nSVM Classification Applied Successfully!")
