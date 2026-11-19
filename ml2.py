
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)

# ==================================
# PART 1: LINEAR REGRESSION
# ==================================

# Load temperature dataset
file = input("Enter temperatures.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# Select monthly temperature columns
month_columns = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
]

# Convert month columns into rows
df = df.melt(
    id_vars=['YEAR'],
    value_vars=month_columns,
    var_name='MONTH',
    value_name='TEMPERATURE'
)

# Remove missing values
df = df.dropna(subset=['YEAR', 'TEMPERATURE'])

# Convert month names into numbers
month_map = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

df['MONTH'] = df['MONTH'].map(month_map)

# Select input and output
X = df[['YEAR', 'MONTH']]
y = df['TEMPERATURE']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
y_pred = lr.predict(X_test)

# Evaluation
print("\nLINEAR REGRESSION RESULTS")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()


# ==================================
# PART 2: LOGISTIC REGRESSION
# ==================================

# Load Social Network Ads dataset
file = input("Enter Social_Network_Ads.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# Remove unnecessary ID column
if 'User ID' in df.columns:
    df = df.drop('User ID', axis=1)

# Remove duplicate records
df = df.drop_duplicates()

# Handle missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['EstimatedSalary'] = df['EstimatedSalary'].fillna(
    df['EstimatedSalary'].median()
)

# Convert Gender into numbers
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Select input and output
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale the data
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

# Prediction
y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 5))
plt.imshow(confusion_matrix(y_test, y_pred), cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.xticks([0, 1], ['Not Purchased', 'Purchased'])
plt.yticks([0, 1], ['Not Purchased', 'Purchased'])
plt.colorbar()
plt.show()

print("\nAssignment Completed Successfully!")
