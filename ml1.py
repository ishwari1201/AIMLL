
import pandas as pd
import matplotlib.pyplot as plt

# Take dataset path from user
file = input("Enter Heart.csv file path: ").strip().strip('"')

# Read dataset
df = pd.read_csv(file)

# Display first five records
print("\nFirst 5 Records:")
print(df.head())

# Display dataset shape
print("\nShape of Dataset:")
print(df.shape)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing numerical values with mean
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Display missing values after preprocessing
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Plot one visualization
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=10, edgecolor='black')
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of Patients")
plt.show()

print("\nData Preprocessing and EDA Completed Successfully!")
