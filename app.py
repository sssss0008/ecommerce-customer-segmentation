import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce Customer Segmentation")
st.markdown("""
This application uses **K-Means Clustering** to segment e-commerce customers based on purchasing behavior and demographics.
""")

st.divider()

# 1. Dataset Generation
@st.cache_data
def generate_customer_data():
    np.random.seed(42)
    n = 120
    g1 = np.column_stack([
        np.random.normal(25, 4, n // 3),
        np.random.normal(35, 8, n // 3),
        np.random.normal(80, 8, n // 3),
        np.random.normal(28, 5, n // 3)
    ])
    g2 = np.column_stack([
        np.random.normal(45, 6, n // 3),
        np.random.normal(105, 12, n // 3),
        np.random.normal(85, 8, n // 3),
        np.random.normal(48, 8, n // 3)
    ])
    g3 = np.column_stack([
        np.random.normal(52, 8, n // 3),
        np.random.normal(65, 15, n // 3),
        np.random.normal(25, 10, n // 3),
        np.random.normal(10, 4, n // 3)
    ])
    
    data = np.vstack([g1, g2, g3])
    df = pd.DataFrame(data, columns=["Age", "Annual Income ($k)", "Spending Score (1-100)", "Total Orders"])
    df["Customer ID"] = [f"CUST-{1000+i}" for i in range(len(df))]
    
    df["Age"] = df["Age"].clip(18, 70).round().astype(int)
    df["Annual Income ($k)"] = df["Annual Income ($k)"].clip(15, 150).round(1)
    df["Spending Score (1-100)"] = df["Spending Score (1-100)"].clip(1, 100).round().astype(int)
    df["Total Orders"] = df["Total Orders"].clip(1, 100).round().astype(int)
    
    return df[["Customer ID", "Age", "Annual Income ($k)", "Spending Score (1-100)", "Total Orders"]]

df_customers = generate_customer_data()
feature_cols = ["Age", "Annual Income ($k)", "Spending Score (1-100)", "Total Orders"]

# Sidebar Controls
st.sidebar.header("⚙️ Model Parameters")
k_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)
use_scaling = st.sidebar.checkbox("Apply Z-Score Standardization", value=True)

X_raw = df_customers[feature_cols].values

if use_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
else:
    X = X_raw.copy()

# Fit K-Means
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df_customers["Cluster"] = kmeans.fit_predict(X)

# 2. Elbow Method
st.subheader("1. Optimal K Evaluation (Elbow Method)")
wcss = []
for k_test in range(1, 8):
    km_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    km_test.fit(X)
    wcss.append(km_test.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(7, 2.8))
ax_elbow.plot(range(1, 8), wcss, marker='o', color='#1C83E1', linewidth=2)
ax_elbow.axvline(x=k_clusters, color='#FF4B4B', linestyle='--', label=f'Selected K={k_clusters}')
ax_elbow.set_xlabel("Number of Clusters (K)")
ax_elbow.set_ylabel("WCSS (Inertia)")
ax_elbow.set_title("Elbow Curve")
ax_elbow.grid(True, linestyle='--', alpha=0.5)
ax_elbow.legend()

col_e1, col_e2 = st.columns([1.2, 1])
with col_e1:
    st.pyplot(fig_elbow)
with col_e2:
    st.markdown("""
    The **Elbow Method** tracks Within-Cluster Sum of Squares (WCSS).
    The point where the line bends sharply indicates the optimal balance between accuracy and cluster count.
    """)

st.divider()

# 3. PCA & Personas
col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    st.subheader("2. Cluster Projection (2D PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig_pca, ax_pca = plt.subplots(figsize=(6, 4.5))
    colors = ['#FF4B4B', '#1C83E1', '#00D4B1', '#FF9F1C', '#9B51E0', '#E83E8C']
    
    for c in range(k_clusters):
        pts = X_pca[df_customers["Cluster"] == c]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c=colors[c % len(colors)], label=f"Segment {c+1}", s=60, alpha=0.7)
        
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', s=180, label='Centroids', edgecolor='white')
    
    ax_pca.set_title("Customer Clusters in 2D Space")
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    ax_pca.legend()
    st.pyplot(fig_pca)

with col_v2:
    st.subheader("3. Segment Averages")
    cluster_means = df_customers.groupby("Cluster")[feature_cols].mean().round(1)
    cluster_counts = df_customers.groupby("Cluster").size().rename("Customer Count")
    summary_df = pd.concat([cluster_counts, cluster_means], axis=1)
    summary_df.index = [f"Segment {i+1}" for i in summary_df.index]
    st.dataframe(summary_df, use_container_width=True)

st.divider()

# 4. Customer Classifier
st.subheader("4. Classify New Customer")
c1, c2, c3, c4 = st.columns(4)
new_age = c1.number_input("Age", 18, 90, 30)
new_income = c2.number_input("Annual Income ($k)", 10, 200, 85)
new_spending = c3.number_input("Spending Score (1-100)", 1, 100, 80)
new_orders = c4.number_input("Total Orders", 1, 100, 35)

new_point_raw = np.array([[new_age, new_income, new_spending, new_orders]])
if use_scaling:
    new_point_scaled = scaler.transform(new_point_raw)
else:
    new_point_scaled = new_point_raw

assigned_seg = kmeans.predict(new_point_scaled)[0]
st.success(f"🎯 Assigned Segment: **Segment {assigned_seg + 1}**")

st.divider()

# 5. Full Data View
st.subheader("5. Customer Dataset")
st.dataframe(df_customers, use_container_width=True)
