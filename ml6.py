
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
file = input("Enter melb_data.csv file path: ").strip().strip('"')
df = pd.read_csv(file)

print("\nFirst 5 Records:")
print(df.head())

# Step 2: Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

# Step 3: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Step 4: Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Step 5: Remove duplicate records
df = df.drop_duplicates()

# Step 6: Handle missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# Step 7: Summary statistics
print("\nSummary Statistics:")
print(df.describe())

print("\nCategorical Column Statistics:")
print(df.describe(include='object'))

# Step 8: Display data types
print("\nData Types:")
print(df.dtypes)

# Step 9: Visualization - House Price Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Price'], bins=30, edgecolor='black')
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of Melbourne House Prices")
plt.show()

print("\nData Preprocessing and EDA Completed Successfully!")
