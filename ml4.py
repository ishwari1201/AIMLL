
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
file = input("Enter CSV file path: ").strip().strip('"')
df = pd.read_csv(file)

df.columns = df.columns.str.strip()

print("\nFirst 5 Records:")
print(df.head())

# Step 2: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Remove missing values
df = df.dropna()

# Step 4: Remove duplicate records
df = df.drop_duplicates()

# Step 5: Drop Customer ID column
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)
elif 'Customer_ID' in df.columns:
    df = df.drop('Customer_ID', axis=1)

# Step 6: Label Encoding
le = LabelEncoder()
if 'Gender' in df.columns:
    df['Gender'] = le.fit_transform(df['Gender'])

# Step 7: Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 8: Feature Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# ---------------- K-MEANS CLUSTERING ----------------

print("\nK-MEANS CLUSTERING")

# Step 9: Create K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Step 10: Train model and predict clusters
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 11: Evaluate K-Means
print("\nK-Means Silhouette Score:")
print(silhouette_score(X_scaled, kmeans_labels))

# Step 12: Visualize K-Means clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmeans_labels
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# ---------------- HIERARCHICAL CLUSTERING ----------------

print("\nHIERARCHICAL CLUSTERING")

# Step 13: Create Hierarchical model
hc = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward'
)

# Step 14: Train model and predict clusters
hc_labels = hc.fit_predict(X_scaled)

# Step 15: Evaluate Hierarchical Clustering
print("\nHierarchical Silhouette Score:")
print(silhouette_score(X_scaled, hc_labels))

# Step 16: Visualize Hierarchical clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=hc_labels
)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Step 17: Conclusion
print("\nK-Means and Hierarchical Clustering Applied Successfully!")
