import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv('D:\DCS\Sem 4\Predictive Analytics\Project\Dataset\Walmart.csv')

# Preprocessing
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['ShipDate'] = pd.to_datetime(df['ShipDate'])
df['Year'] = df['OrderDate'].dt.year
df['Month'] = df['OrderDate'].dt.month

# Clustering
features = ['Sales', 'Quantity', 'Profit']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Time Series Analysis
time_series_df = df.set_index('OrderDate')['Sales']

# Check for duplicate labels
if time_series_df.index.has_duplicates:
    time_series_df = time_series_df.groupby(level=0).mean()

# Set the frequency of the time series data
time_series_df = time_series_df.asfreq('D')

decomposition = seasonal_decompose(time_series_df, model='additive')
trend = decomposition.trend
seasonality = decomposition.seasonal
residuals = decomposition.resid

# Plotting
# Cluster Analysis
plt.figure(figsize=(12, 6))
plt.scatter(df['PC1'], df['PC2'], c=df['Cluster'])
plt.title('Clustering of Walmart Sales Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# PCA Analysis
plt.figure(figsize=(8, 6))
plt.scatter(df['PC1'], df['PC2'])
plt.title('PCA Analysis of Walmart Sales Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Time Series Analysis
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time_series_df, label='Original')
plt.legend(loc='best')
plt.title('Walmart Sales Time Series Data')
plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(4, 1, 3)
plt.plot(seasonality, label='Seasonality')
plt.legend(loc='best')
plt.subplot(4, 1, 4)
plt.plot(residuals, label='Residuals')
plt.legend(loc='best')
plt.show()
