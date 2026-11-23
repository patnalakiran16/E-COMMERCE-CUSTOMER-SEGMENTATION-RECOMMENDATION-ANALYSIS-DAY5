Day 45: E-Commerce Customer Segmentation & Recommendation Analysis

📌 Project Overview

This project focuses on analysing e-commerce customer purchasing behaviour using Machine Learning techniques.

The main objective is to identify different customer groups based on their purchasing patterns and develop a basic product recommendation system using Cosine Similarity.

Customer segmentation is performed using the K-Means Clustering algorithm, while RFM (Recency, Frequency, Monetary) analysis helps understand customer behaviour.

🎯 Project Objectives

- Analyse customer purchasing behaviour.
- Clean and preprocess transaction data.
- Perform RFM analysis.
- Apply K-Means clustering for customer segmentation.
- Identify the optimal number of clusters using the Elbow Method.
- Evaluate clustering using Silhouette Score.
- Visualize customer segments in 2D.
- Build a basic product recommendation system.
- Generate business recommendations for each customer segment.

🛠️ Technologies & Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

📂 Dataset

Dataset Name: Online Retail Dataset

Source: UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains online retail transaction records, including:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

⚙️ Project Workflow

1. Data Collection

Loaded the Online Retail dataset using Pandas.

2. Data Cleaning

- Removed missing customer IDs.
- Removed missing product descriptions.
- Removed cancelled transactions.
- Removed invalid quantities and prices.
- Calculated total transaction amount.

3. Exploratory Data Analysis (EDA)

- Analysed total customers.
- Identified unique products.
- Calculated total revenue.
- Visualized top-selling products.

4. RFM Analysis

Created three important customer behaviour features:

- Recency: Number of days since the customer's last purchase.
- Frequency: Number of unique invoices purchased.
- Monetary: Total amount spent by the customer.

5. Feature Scaling

Applied StandardScaler to normalize customer features before clustering.

Log transformation was also applied to reduce skewness.

6. K-Means Clustering

Used K-Means clustering to group customers based on their purchasing behaviour.

7. Cluster Evaluation

Used the following methods:

- Elbow Method
- Silhouette Score

8. Customer Segmentation

Identified four business-friendly customer segments:

- Champions
- Regular Customers
- Occasional Buyers
- At-Risk Customers

9. Customer Visualization

Used PCA to reduce customer features into two dimensions and visualize different customer clusters.

10. Product Recommendation System

Built a basic recommendation system using:

- Customer-Product Purchase Matrix
- Cosine Similarity
- Similar Customer Analysis

The system recommends products based on the purchasing behaviour of similar customers.

📊 Business Recommendations

Customer Segment| Business Strategy
Champions| VIP offers and loyalty rewards
Regular Customers| Repeat purchase discounts
Occasional Buyers| Promotional campaigns
At-Risk Customers| Re-engagement offers

📈 Project Deliverables

The project generates the following output files:

1. "customer_segmentation_results.csv"
2. "customer_segment_profiles.csv"
3. "product_recommendations.csv"
4. "cleaned_online_retail.csv"

▶️ How to Run the Project

Step 1: Clone the repository.

git clone YOUR_GITHUB_REPOSITORY_URL

Step 2: Navigate to the project folder.

cd E-Commerce-Customer-Segmentation

Step 3: Install required libraries.

pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

Step 4: Download the Online Retail dataset and place "Online Retail.xlsx" in the project folder.

Step 5: Open Jupyter Notebook.

jupyter notebook

Step 6: Open the Python notebook and run all cells.

💡 Key Learnings

- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- RFM customer analysis.
- Unsupervised Machine Learning.
- K-Means clustering.
- Cluster evaluation techniques.
- Data visualization using PCA.
- Cosine Similarity.
- Building a basic recommendation system.
- Business-oriented data analysis.

🏁 Conclusion

This project demonstrates how Machine Learning can be used to understand customer purchasing behaviour, identify valuable customer segments, and recommend relevant products.

Customer segmentation helps businesses improve customer engagement, develop targeted marketing strategies, and make data-driven decisions.

---

👨‍💻 Author

Patnala Kiran

AI & ML Intern | MCA Student

Project: Day 45 – E-Commerce Customer Segmentation & Recommendation Analysis

Skills: Python | Machine Learning | Data Analysis | Pandas | Scikit-learn

#Python #MachineLearning #CustomerSegmentation #KMeansClustering #DataAnalysis #RecommendationSystem #Intern