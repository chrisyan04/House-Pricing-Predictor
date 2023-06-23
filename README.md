# House Pricing Predictor
The purpose for this project is to learn about machine learning with Python and its libraries. 
Here the task is to produce and exploratory analysis and build predictive models for our housing dataset 
obtained from Kaggle.

View the California Housing Prices datset here: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Report Summary
### 1. Data Loading and Cleaning:
- The housing dataset is loaded into a pandas DataFrame.
- The information about the dataset is displayed, showing the columns and their data types.
- The dataset contains missing values in the 'total_bedrooms' column, which are dropped using the `dropna()` function.
### 2. Data Exploration and Visualization:
- Train-test split is performed on the cleaned dataset using a test size of 20%.
- Histograms are plotted to visualize the distribution of various features in the training data.
- A correlation matrix heatmap is created using `sns.heatmap()` to examine the relationships between features.
