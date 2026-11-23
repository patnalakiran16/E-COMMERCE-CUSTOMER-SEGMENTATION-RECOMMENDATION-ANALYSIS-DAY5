# ============================================================
# DAY 45: E-COMMERCE CUSTOMER SEGMENTATION & RECOMMENDATION
# ============================================================

# Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")


# Step 2: Load Dataset

file_path = "Online Retail.xlsx"

df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")
print(df.head())


# Step 3: Understand Dataset

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# Step 4: Data Cleaning

df = df.dropna(subset=[
    "CustomerID",
    "Description",
    "InvoiceDate"
])

# Remove cancelled transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid quantities and prices
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

df["CustomerID"] = df["CustomerID"].astype(int)

# Calculate total transaction value
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("\nCleaned Dataset Shape:", df.shape)


# Step 5: Exploratory Data Analysis

print("\nTotal Customers:", df["CustomerID"].nunique())

print("Total Products:", df["StockCode"].nunique())

print("Total Revenue:", round(df["TotalAmount"].sum(), 2))

# Top 10 Selling Products
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


# Step 6: Create RFM Features

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({

    "InvoiceDate": lambda x: (reference_date - x.max()).days,

    "InvoiceNo": "nunique",

    "TotalAmount": "sum"

})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nRFM Analysis:")
print(rfm.head())

print("\nRFM Statistics:")
print(rfm.describe())


# Step 7: Handle Skewness

rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()

rfm_features = np.log1p(rfm_features)


# Step 8: Feature Scaling

scaler = StandardScaler()

scaled_features = scaler.fit_transform(rfm_features)

print("\nFeature Scaling Completed!")


# Step 9: Find Optimal Clusters Using Elbow Method

inertia_values = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_features)

    inertia_values.append(model.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# Step 10: Silhouette Analysis

sample_size = min(2000, len(scaled_features))

silhouette_scores = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_features)

    score = silhouette_score(
        scaled_features,
        labels,
        sample_size=sample_size,
        random_state=42
    )

    silhouette_scores.append(score)

    print(f"Clusters: {k}, Silhouette Score: {score:.4f}")


# Step 11: Apply K-Means Clustering

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nCustomer Clustering Completed!")

print(rfm.head())


# Step 12: Create Customer Segment Names

cluster_profile = rfm.groupby("Cluster").agg({

    "Recency": "mean",

    "Frequency": "mean",

    "Monetary": "mean"

})

# High frequency and monetary value with low recency
cluster_profile["SegmentScore"] = (
    cluster_profile["Frequency"].rank(pct=True) +
    cluster_profile["Monetary"].rank(pct=True) -
    cluster_profile["Recency"].rank(pct=True)
)

sorted_clusters = cluster_profile["SegmentScore"].sort_values(
    ascending=False
).index.tolist()

segment_names = [
    "Champions",
    "Regular Customers",
    "Occasional Buyers",
    "At-Risk Customers"
]

segment_mapping = dict(zip(sorted_clusters, segment_names))

rfm["CustomerSegment"] = rfm["Cluster"].map(segment_mapping)

print("\nCustomer Segment Distribution:")
print(rfm["CustomerSegment"].value_counts())


# Step 13: Customer Segment Visualization

plt.figure(figsize=(9, 5))

sns.countplot(
    data=rfm,
    x="CustomerSegment",
    order=segment_names
)

plt.title("Customer Segmentation Distribution")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# Step 14: 2D Customer Cluster Visualization

pca = PCA(n_components=2)

reduced_features = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(
    reduced_features,
    columns=["Component 1", "Component 2"],
    index=rfm.index
)

pca_df["Cluster"] = rfm["Cluster"].values

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=pca_df,
    x="Component 1",
    y="Component 2",
    hue="Cluster",
    palette="viridis",
    s=40
)

plt.title("2D Customer Cluster Visualization")
plt.tight_layout()
plt.show()


# Step 15: Business-Friendly Segment Profiles

segment_summary = rfm.groupby("CustomerSegment").agg({

    "Recency": "mean",

    "Frequency": "mean",

    "Monetary": "mean",

    "CustomerID": "count"

}).rename(columns={

    "CustomerID": "CustomerCount"

})

print("\nBusiness-Friendly Customer Profiles:")
print(segment_summary.round(2))


# Step 16: Create Customer-Product Purchase Matrix

customer_product = df.pivot_table(

    index="CustomerID",

    columns="StockCode",

    values="Quantity",

    aggfunc="sum",

    fill_value=0

)

print("\nCustomer-Product Matrix Shape:", customer_product.shape)


# Step 17: Apply Cosine Similarity

purchase_matrix = csr_matrix(customer_product.values)

similarity_matrix = cosine_similarity(
    purchase_matrix,
    dense_output=False
)

print("\nCosine Similarity Completed!")


# Step 18: Product Recommendation Function

def recommend_products(customer_id, top_n=5):

    if customer_id not in customer_product.index:

        print("Customer ID not found!")

        return pd.DataFrame()

    customer_position = customer_product.index.get_loc(customer_id)

    similarity_scores = similarity_matrix.getrow(
        customer_position
    ).toarray().flatten()

    # Find similar customers
    similar_positions = np.argsort(
        similarity_scores
    )[::-1]

    similar_positions = [
        pos for pos in similar_positions
        if pos != customer_position
        and similarity_scores[pos] > 0
    ][:10]

    if len(similar_positions) == 0:

        print("No similar customers found!")

        return pd.DataFrame()

    # Calculate recommendation scores
    similar_weights = similarity_scores[similar_positions]

    product_scores = np.asarray(
        purchase_matrix[similar_positions].T.dot(similar_weights)
    ).flatten()

    # Exclude products already purchased
    purchased = np.asarray(
        customer_product.iloc[customer_position].values
    )

    product_scores[purchased > 0] = -1

    # Get top recommended products
    recommended_positions = np.argsort(
        product_scores
    )[::-1]

    recommended_positions = [
        pos for pos in recommended_positions
        if product_scores[pos] >= 0
    ][:top_n]

    recommended_codes = customer_product.columns[
        recommended_positions
    ]

    # Get product descriptions
    product_details = (
        df[["StockCode", "Description"]]
        .drop_duplicates("StockCode")
        .set_index("StockCode")
    )

    recommendations = pd.DataFrame({

        "StockCode": recommended_codes,

        "RecommendationScore": [
            product_scores[pos]
            for pos in recommended_positions
        ]

    })

    recommendations["Description"] = (
        recommendations["StockCode"]
        .map(product_details["Description"])
    )

    return recommendations[
        ["StockCode", "Description", "RecommendationScore"]
    ]


# Step 19: Test Product Recommendation

sample_customer = rfm.index[0]

print("\nSample Customer ID:", sample_customer)

print("\nRecommended Products:")

recommendations = recommend_products(
    sample_customer,
    top_n=5
)

print(recommendations)


# Step 20: Business Recommendations for Each Segment

business_recommendations = {

    "Champions":
    "Provide VIP offers, loyalty rewards and early product access.",

    "Regular Customers":
    "Encourage repeat purchases through loyalty points and discounts.",

    "Occasional Buyers":
    "Use promotional offers and personalized email campaigns.",

    "At-Risk Customers":
    "Send re-engagement offers and special discounts to bring them back."

}

print("\nBusiness Recommendations:")

for segment, recommendation in business_recommendations.items():

    print(f"\n{segment}:")
    print(recommendation)


# Step 21: Save Project Deliverables

rfm.to_csv(
    "customer_segmentation_results.csv"
)

segment_summary.to_csv(
    "customer_segment_profiles.csv"
)

recommendations.to_csv(
    "product_recommendations.csv",
    index=False
)

# Save cleaned dataset
df.to_csv(
    "cleaned_online_retail.csv",
    index=False
)

print("\nAll Project Files Saved Successfully!")


# Step 22: Final Project Summary

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("===================================")

print("Total Customers:", len(rfm))

print("Total Segments:", rfm["CustomerSegment"].nunique())

print("Total Products:", df["StockCode"].nunique())

print("Total Revenue:", round(df["TotalAmount"].sum(), 2))

print("\nCustomer Segments:")

print(rfm["CustomerSegment"].value_counts())

print("\nThank You!")
