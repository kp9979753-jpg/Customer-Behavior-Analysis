import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================================
# STEP 1 & 2: DATA LOADING & BASIC INSPECTION
# =========================================================================
print("Step 1 & 2: Data Load ho raha hai...")
df = pd.read_csv("ecommerce_customer_data_large.csv")

print("\nInitial Columns in Dataset:")
print(df.columns.tolist())


# =========================================================================
# STEP 3: DATA CLEANING (Missing Values & Duplicates Handle)
# =========================================================================
print("\nStep 3: Data Cleaning shuru ho gayi hai...")

# 1. Returns column ke null values ko 0 se fill karein
df['Returns'] = df['Returns'].fillna(0)

# 2. Duplicate 'Age' column ko drop karein agar wo exist karta hai
if 'Age' in df.columns:
    df = df.drop(columns=['Age'])

# 3. Purchase Date ko text se datetime format mein convert karein
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

print("Data successfully cleaned! Koi missing value nahi bachi.")


# =========================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (Graphs & Visualizations)
# =========================================================================
print("\nStep 4: Graphs generate ho rahe hain...")

# --- GRAPH 1: Product Category Distribution ---
plt.figure(figsize=(8, 4)) 
sns.countplot(data=df, x='Product Category', palette='pastel')
plt.title('Product Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')

# --- GRAPH 2: Monthly Sales Trends ---
df['Month'] = df['Purchase Date'].dt.to_period('M').astype(str)
monthly_sales = df.groupby('Month')['Total Purchase Amount'].sum().reset_index()

plt.figure(figsize=(12, 5)) 
sns.lineplot(data=monthly_sales, x='Month', y='Total Purchase Amount', marker='o', color='green')
plt.xticks(rotation=45)
plt.title('Monthly Sales Trends')
plt.grid(True)


# =========================================================================
# STEP 5: CHURN RISK ANALYSIS (Advanced Analysis)
# =========================================================================
print("\nStep 5: Churn Risk ki janch ho rahi hai...")

# --- GRAPH 3: Churn Risk by Gender ---
plt.figure(figsize=(6, 4)) 
sns.countplot(data=df, x='Gender', hue='Churn', palette='Set1')
plt.title('Churn Risk by Gender')

# Print Churn Rates in Terminal
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print(f"Active Customers (0): {churn_rate[0]:.2f}%")
print(f"Churned Customers (1): {churn_rate[1]:.2f}%")


# =========================================================================
# STEP 6: CUSTOMER SEGMENTATION (RFM Features)
# =========================================================================
print("\nStep 6: Customer Segmentation (RFM Analysis) shuru...")

# Recency ke liye base date nikalna
snapshot_date = df['Purchase Date'].max() + pd.Timedelta(days=1)

# Recency, Frequency, aur Monetary calculate karna
rfm = df.groupby('Customer ID').agg({
    'Purchase Date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Customer ID': 'count',                                     # Frequency
    'Total Purchase Amount': 'sum'                              # Monetary
})

# Columns ke naam sahi karna
rfm.columns = ['Recency', 'Frequency', 'Monetary']
print("RFM Table Head:")
print(rfm.head())


# =========================================================================
# STEP 7: MACHINE LEARNING (K-Means Clustering)
# =========================================================================
print("\nStep 7: Machine Learning Model (K-Means) chal raha hai...")

# 1. Data ko scale (normalize) karna zaroori hai model ke liye
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# 2. Customers ko 3 segments/groups mein baantna
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\nSegmentation Complete! Har customer ko group mil gaya hai.")
print(rfm['Cluster'].value_counts())

# --- GRAPH 4: Customer Segments (ML Output Visual) ---
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Customer Segments using K-Means Clustering')


# =========================================================================
# STEP 8: SUMMARY & RECOMMENDATIONS (Terminal Insights)
# =========================================================================
print("\n" + "="*50)
print("STEP 8: KEY OBSERVATIONS & RECOMMENDATIONS FOR INTERNSHIP REPORT")
print("="*50)
print("1. Top product categories ko marketing mein push karein.")
print("2. Monthly trends ke anusar slow months mein zyada discounts/offers dein.")
print("3. Churn aur returns ko control karne ke liye customer feedback lena shuru karein.")
print("4. Clusters (Groups) ke hisab se high-value customers ko loyalty rewards dein.")
print("="*50)


# =========================================================================
# POORE PROJECT KE SAARE GRAPHS EK SAATH SHOW KARNE KE LIYA
# =========================================================================
print("\nTeeno Graphs aur ML ka 4th Graph ek saath popup windows mein open ho rahe hain...")
plt.show()