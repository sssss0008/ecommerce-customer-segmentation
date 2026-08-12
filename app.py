import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Helper Functions ---
def format_npr(amount):
    """Formats numeric values into standard Nepalese Rupee string format."""
    return f"NPR {amount:,.2f}"

# --- Page Configuration ---
st.set_page_config(
    page_title="E-Commerce Price Prediction & Recommendation (NPR)",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce Price Prediction & Recommendation Engine")
st.markdown("""
This dual-engine application uses **Random Forest Regression** to estimate optimal product listing prices in **Nepalese Rupees (NPR)** and **K-Means Clustering** to recommend similar catalog items.
""")

st.divider()

# --- 1. Dataset Generation (NPR Scale) ---
@st.cache_data
def load_ecommerce_data():
    np.random.seed(42)
    n = 250
    
    categories = np.random.choice(["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Groceries & Snacks"], size=n)
    brand_tiers = np.random.choice(["Budget", "Mid-Range", "Premium"], size=n, p=[0.4, 0.4, 0.2])
    ratings = np.round(np.random.uniform(2.5, 5.0, size=n), 1)
    review_counts = np.random.randint(10, 3000, size=n)
    feature_scores = np.random.randint(20, 100, size=n)
    
    # Manufacturing Cost generated in NPR (Range: NPR 200 to NPR 35,000)
    mfg_costs = np.random.uniform(200, 35000, size=n)
    
    category_multiplier = {
        "Electronics": 1.45, 
        "Fashion": 1.75, 
        "Home & Kitchen": 1.35, 
        "Beauty": 2.10,
        "Groceries & Snacks": 1.25
    }
    brand_multiplier = {"Budget": 1.20, "Mid-Range": 1.60, "Premium": 2.50}
    
    prices = []
    for cat, brand, cost, rating, feat in zip(categories, brand_tiers, mfg_costs, ratings, feature_scores):
        base_price = cost * category_multiplier[cat] * brand_multiplier[brand]
        bonus = (rating * 150) + (feat * 12)
        price = base_price + bonus + np.random.normal(0, 300)
        prices.append(max(round(price, 2), round(cost * 1.15, 2)))
        
    df = pd.DataFrame({
        "Product ID": [f"PROD-NP-{1000+i}" for i in range(n)],
        "Category": categories,
        "Brand Tier": brand_tiers,
        "Mfg Cost (NPR)": np.round(mfg_costs, 2),
        "Rating": ratings,
        "Review Count": review_counts,
        "Feature Score": feature_scores,
        "Price (NPR)": prices
    })
    return df

df_products = load_ecommerce_data()

# --- 2. Model Training (Price Predictor) ---
categorical_features = ["Category", "Brand Tier"]
numerical_features = ["Mfg Cost (NPR)", "Rating", "Review Count", "Feature Score"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

regressor_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X = df_products[categorical_features + numerical_features]
y = df_products["Price (NPR)"]
regressor_pipeline.fit(X, y)

# --- 3. Clustering Model (Recommendation Engine) ---
X_processed = preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_products["Cluster"] = kmeans.fit_predict(X_processed)

# --- Sidebar Controls (Predictor Input in NPR) ---
st.sidebar.header("📥 Product Details Input")

input_category = st.sidebar.selectbox("Category", ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Groceries & Snacks"])
input_brand = st.sidebar.selectbox("Brand Tier", ["Budget", "Mid-Range", "Premium"])
input_mfg_cost = st.sidebar.number_input("Manufacturing Cost (NPR)", min_value=50.0, max_value=100000.0, value=3500.0, step=100.0)
input_rating = st.sidebar.slider("Expected Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
input_reviews = st.sidebar.number_input("Expected Review Count", min_value=0, max_value=5000, value=250)
input_feat_score = st.sidebar.slider("Feature Quality Score (1-100)", min_value=1, max_value=100, value=65)

# --- Main Tabs ---
tab1, tab2 = st.tabs(["💰 NPR Price Prediction", "🎯 Product Recommendations"])

# TAB 1: PRICE PREDICTION
with tab1:
    st.subheader("1. Optimal Listing Price Estimation")
    
    input_data = pd.DataFrame([{
        "Category": input_category,
        "Brand Tier": input_brand,
        "Mfg Cost (NPR)": input_mfg_cost,
        "Rating": input_rating,
        "Review Count": input_reviews,
        "Feature Score": input_feat_score
    }])
    
    predicted_price = regressor_pipeline.predict(input_data)[0]
    estimated_margin = predicted_price - input_mfg_cost
    margin_pct = (estimated_margin / predicted_price) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Price", format_npr(predicted_price))
    m2.metric("Estimated Profit Margin", format_npr(estimated_margin))
    m3.metric("Margin Ratio", f"{margin_pct:.1f}%")
    
    st.divider()
    
    st.subheader("2. Price Benchmark in Category")
    cat_df = df_products[df_products["Category"] == input_category]
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(cat_df["Price (NPR)"], bins=15, color='#1C83E1', alpha=0.7, edgecolor='black')
    ax.axvline(predicted_price, color='#FF4B4B', linestyle='--', linewidth=2, label=f'Predicted: {format_npr(predicted_price)}')
    ax.set_title(f"Market Price Distribution for {input_category}")
    ax.set_xlabel("Price in NPR")
    ax.set_ylabel("Product Count")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig)

# TAB 2: PRODUCT RECOMMENDATIONS
with tab2:
    st.subheader("1. In-Cluster Item Recommendations")
    
    user_processed = preprocessor.transform(input_data)
    assigned_cluster = kmeans.predict(user_processed)[0]
    
    st.info(f"Your custom product matches **Cluster {assigned_cluster + 1}** segment characteristics.")
    
    cluster_products = df_products[df_products["Cluster"] == assigned_cluster].copy()
    cluster_products["Price Difference (NPR)"] = np.abs(cluster_products["Price (NPR)"] - predicted_price)
    recommendations = cluster_products.sort_values(by="Price Difference (NPR)").head(5).copy()
    
    # Format currency columns for rendering
    display_recs = recommendations.copy()
    display_recs["Price (NPR)"] = display_recs["Price (NPR)"].apply(format_npr)
    display_recs["Mfg Cost (NPR)"] = display_recs["Mfg Cost (NPR)"].apply(format_npr)
    
    st.markdown("### Top Recommended Similar Products")
    st.dataframe(
        display_recs[["Product ID", "Category", "Brand Tier", "Rating", "Feature Score", "Mfg Cost (NPR)", "Price (NPR)"]],
        use_container_width=True
    )
    
    st.divider()
    
    st.subheader("2. Product Mapping (2D PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    user_pca = pca.transform(user_processed)
    
    fig_pca, ax_pca = plt.subplots(figsize=(8, 4))
    colors = ['#FF4B4B', '#1C83E1', '#00D4B1', '#FF9F1C']
    
    for c in range(4):
        pts = X_pca[df_products["Cluster"] == c]
        ax_pca.scatter(pts[:, 0], pts[:, 1], c=colors[c], label=f"Cluster {c+1}", s=40, alpha=0.6)
        
    ax_pca.scatter(user_pca[:, 0], user_pca[:, 1], c='black', marker='*', s=250, label='Target Item', edgecolor='white')
    ax_pca.set_title("Product Feature Space Projection")
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    ax_pca.legend()
    st.pyplot(fig_pca)

st.divider()

# --- Full Dataset View with NPR formatting ---
st.subheader("📋 Full Catalog Overview")
df_display = df_products.copy()
df_display["Price (NPR)"] = df_display["Price (NPR)"].apply(format_npr)
df_display["Mfg Cost (NPR)"] = df_display["Mfg Cost (NPR)"].apply(format_npr)
st.dataframe(df_display, use_container_width=True)
