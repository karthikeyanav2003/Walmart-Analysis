# Walmart-Analysis
ABSTRACT:
The project performs an analysis of sales data from Walmart using various techniques. It 
includes exploratory data analysis, dimensionality reduction using principal component 
analysis (PCA), clustering, and time series analysis for demand forecasting.
PCA:
• Numeric variables (Sales, Quantity, Profit) are selected for PCA.
• The variables are scaled, and PCA is performed to obtain the principal 
components.
• The variance explained by each principal component is calculated and printed.
Cluster ANALYSIS:
• The optimal number of clusters is determined using the silhouette score, which 
measures the quality of clustering solutions.
• The silhouette scores for different values of k are calculated and plotted.
• Clustering is performed using k-means with the optimal number of clusters.
• The cluster assignments and a visualization of the clusters are displayed.
TIME SERIES ANALYSIS AND FORECASTING:
• The sales data is aggregated by date to obtain daily sales.
• A time series object is created with a frequency of 7 days.
• The time series is decomposed into trend, seasonal, and random components, 
which are plotted.
• An ARIMA model is automatically fitted to the time series data.
Sales are forecasted for the next 7 days using the ARIMA model, and the forecasted values 
are plotted.
