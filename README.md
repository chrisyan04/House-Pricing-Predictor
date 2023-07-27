# House Pricing Predictor 🌆

## Purpose ❓
The purpose for this project is to learn about machine learning with Python and its libraries. 
Here the task is to produce and exploratory analysis and build predictive models for our housing dataset 
obtained from Kaggle.

View the California Housing Prices datset here: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Report Process 📊
### 1. Data Loading and Cleaning:
- The housing dataset is loaded into a pandas DataFrame.
- The information about the dataset is displayed, showing the columns and their data types.
- The dataset contains missing values in the 'total_bedrooms' column, which are dropped using the `dropna()` function.
### 2. Data Exploration and Visualization:
- Train-test split is performed on the cleaned dataset using a test size of 20%.
- Histograms are plotted to visualize the distribution of various features in the training data.
- A correlation matrix heatmap is created using `sns.heatmap()` to examine the relationships between features.
### 3. Feature Engineering:
- The code applies logarithmic transformation to the 'total_rooms', 'total_bedrooms', 'population', and 'households' columns to adjust their distributions.
- Another set of histograms is plotted to visualize the transformed features.
- The 'ocean_proximity' column is one-hot encoded using `pd.get_dummies()` and then dropped from the dataset.
### 4. Linear Regression Model:
- The dataset is prepared for training a linear regression model by separating the target variable ('median_house_value') from the input features.
- The input features are standardized using `StandardScaler()` to ensure their scales are comparable.
- A linear regression model is created and trained using the standardized input features and target variable.
### 5. Model Evaluation:
- The test data is preprocessed similarly to the training data, including logarithmic transformation and one-hot encoding.
- The trained linear regression model is evaluated on the preprocessed test data using the `score()` function, which computes the coefficient of determination (R-squared) between predicted and actual target values.
### 6. Random Forest model:
- A random forest regressor model is instantiated and trained on the standardized training data.
- The trained random forest model is evaluated on the preprocessed test data, similar to the linear regression model.
### 7. Hyperparameter Tuning:
- Grid search with cross-validation is performed using `GridSearchCV` to find the best hyperparameters for the random forest model.
- The specified parameter grid includes different values for 'n_estimators', 'min_samples_split', and 'max_depth'.
- The best estimator is obtained using `best_estimator_`, and its performance is evaluated on the test data.

## Summary 📝
The California Housing Prices project aims to perform an exploratory analysis and develop predictive models using machine learning techniques within Python. The initial steps involve loading and cleaning the housing dataset, followed by data exploration and visualization. Feature engineering techniques are applied, including logarithmic transformation and one-hot encoding. Subsequently, both linear regression and random forest regression models are trained and evaluated on the preprocessed data. Additionally, hyperparameter tuning is performed for the random forest model using grid search and cross-validation. The project provides insights into feature relationships, identitfies important predictors, and assesses the predictive performance of the models. It serves as a learning exercise for machine learning with Python and showcases techniques for data preprocessing, model training/testing, and evaluation. Results obtained from this model, in having a score of around 0.814281, the score being the coefficient of determination of R-squared. Such score is achieved by the best random forest regressor on the test data. It indicates how well our model fits the test data, and since we have a relatively high value, it implies that it is a good fit. Through the process of training with random forest model and also hyperparameter tuning, we bumped the score from around 67.06% to around 81.44%. So our model explains approximately 81.44% of the variance in the target variable based on the test data.

## Technologies Used ⚙️
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 
