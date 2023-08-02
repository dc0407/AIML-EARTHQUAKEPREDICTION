# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# II. Data Collection and Preprocessing
# A. Data Sources
dataset1 = pd.read_csv(r"C:\Users\likhith sri sai\Downloads\19ECE363 - Machine Learning Project\earthquake_data_1.csv")
dataset2 = pd.read_csv(r"C:\Users\likhith sri sai\Downloads\19ECE363 - Machine Learning Project\earthquake_data_2.csv")
dataset3 = pd.read_csv(r"C:\Users\likhith sri sai\Downloads\19ECE363 - Machine Learning Project\earthquake_data_3.csv")

# B. Data Description
# Combine the datasets
all_datasets = [dataset1, dataset2, dataset3]
df = pd.concat(all_datasets)

# C. Data Cleaning
# 1. Outlier Removal
z_scores = np.abs((df['magnitude'] - df['magnitude'].mean()) / df['magnitude'].std())
df = df[(z_scores < 3)]

# 2. Null Value Handling
df = df.dropna()

# D. Feature Engineering
# 1. Datetime Conversion
df['date/time'] = pd.to_datetime(df['date/time'])

# 2. Creation of Numerical Features
df['year'] = df['date/time'].dt.year
df['month'] = df['date/time'].dt.month
df['day'] = df['date/time'].dt.day
df['hour'] = df['date/time'].dt.hour
df['minute'] = df['date/time'].dt.minute
df['second'] = df['date/time'].dt.second

# E. Feature Scaling
scaler = StandardScaler()
df[['latitude', 'longitude', 'depth','year', 'month','day', 'hour', 'minute', 'second']] = scaler.fit_transform(df[['latitude', 'longitude', 'depth','year', 'month','day', 'hour', 'minute', 'second']])

# III. Exploratory Data Analysis
# A. Data Visualization
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
sns.histplot(data=df, x="latitude", ax=axs[0, 0])
sns.histplot(data=df, x="longitude", ax=axs[0, 1])
sns.histplot(data=df, x="depth", ax=axs[1, 0])
sns.histplot(data=df, x="magnitude", ax=axs[1, 1])
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="magnitude")
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="year", y="magnitude")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="longitude", y="latitude", hue="magnitude")
plt.show()

# B. Statistical Parameters Calculation
mean_magnitude = df['magnitude'].mean()
std_magnitude = df['magnitude'].std()
quartiles = df['magnitude'].quantile([0.25, 0.5, 0.75])
print("Mean of magnitude: ",round(mean_magnitude,2))
print("Standard deviation: ",round(std_magnitude,2))
print("Interquartile ranges: ",quartiles)

# IV. Correlation Analysis
# A. Correlation Matrix
corr_matrix = df.corr()

# B. Heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# V. Principal Component Analysis (PCA)
# A. Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour', 'minute', 'second']])

# B. Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print explained variance ratio
print('Explained Variance Ratio:', explained_variance_ratio)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.show()

# VI. Model Development and Evaluation
# A. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df[['latitude', 'longitude', 'depth', 'year', 'month','day', 'hour', 'minute', 'second']], df['magnitude'], test_size=0.2, random_state=42)

# B. Model 1: K-Nearest Neighbors (K-NN)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

# C. Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

# D. Model 3: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

# VII. Model Performance Evaluation
# A. Performance Metrics
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_mae = mean_absolute_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)

print('K-NN Regression Performance Metrics:')
print('MSE:', round(knn_mse,2))
print('MAE:', round(knn_mae,2))
print('R^2:', round(knn_r2,2))

rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print('Random Forest Regression Performance Metrics:')
print('MSE:', round(knn_mse,2))
print('MAE:', round(knn_mae,2))
print('R^2:', round(knn_r2,2))

lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

print('Linear Regression Performance Metrics:')
print('MSE:', round(lr_mse,2))
print('MAE:', round(lr_mae,2))
print('R^2:', round(lr_r2,2))

# B. Predicted Magnitude and Probability Calculation
knn_predict_input = scaler.transform([[35.6895, 139.6917, 10, 2023, 3, 28, 23, 48, 0]])
knn_mag = knn.predict(knn_predict_input)[0]
knn_prob = np.sum(knn.predict(X_test) >= knn_mag) / len(X_test)

rf_predict_input = scaler.transform([[35.6895, 139.6917, 10, 2023, 3, 28, 23, 48, 0]])
rf_mag = rf.predict(rf_predict_input)[0]
rf_prob = np.sum(rf.predict(X_test) >= rf_mag) / len(X_test)

lr_predict_input = scaler.transform([[139.6917, 37.5538, 10, 2023, 3, 28, 23, 48, 0]])
lr_mag = lr.predict(lr_predict_input)[0]
lr_prob = np.sum(lr.predict(X_test) >= lr_mag) / len(X_test)

# Print the predicted magnitude and probability for each model
print("K-NN Model:")
print("Predicted Magnitude: ", round(knn_mag,2))
print("Predicted Probability: ", round(knn_prob,2))

print("Random Forest Model:")
print("Predicted Magnitude: ", round(rf_mag,2))
print("Predicted Probability: ", round(rf_prob,2))

print("Linear Regression Model:")
print("Predicted Magnitude: ", round(lr_mag,2))
print("Predicted Probability: ", round(lr_prob,2))

# C. Comparison of Model Performance
# 1. Scatter plot of predicted vs. actual magnitudes for K-NN
plt.figure(figsize=(8, 6))
plt.scatter(y_test, knn_y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude (K-NN)')
plt.title('Actual vs. Predicted Magnitude (K-NN)')
plt.show()

# 2. Scatter plot of predicted vs. actual magnitudes for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude (Random Forest)')
plt.title('Actual vs. Predicted Magnitude (Random Forest)')
plt.show()

# 3. Scatter plot of predicted vs. actual magnitudes for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude (Linear Regression)')
plt.title('Actual vs. Predicted Magnitude (Linear Regression)')
plt.show()

# 4. Bar plot of model probabilities
models = ['K-NN', 'Random Forest', 'Linear Regression']
probabilities = [knn_prob, rf_prob, lr_prob]

plt.figure(figsize=(8, 6))
plt.bar(models, probabilities)
plt.xlabel('Model')
plt.ylabel('Probability')
plt.title('Model Probability of Predicting Equal or Higher Magnitude')
plt.show()

# D. Classification Metrics
# Calculate classification metrics for each model
top_percentile_threshold = np.percentile(y_test, 20)
knn_top_percentile_threshold = np.percentile(knn_y_pred, 20)
rf_top_percentile_threshold = np.percentile(rf_y_pred, 20)
lr_top_percentile_threshold = np.percentile(lr_y_pred, 20)
y_test_class = np.where(y_test >= top_percentile_threshold, 1, 0)
knn_y_pred_class = np.where(knn_y_pred >= knn_top_percentile_threshold, 1, 0)
rf_y_pred_class = np.where(rf_y_pred >= rf_top_percentile_threshold, 1, 0)
lr_y_pred_class = np.where(lr_y_pred >= lr_top_percentile_threshold, 1, 0)

knn_accuracy = accuracy_score(y_test_class, knn_y_pred_class)
knn_error_rate = 1 - knn_accuracy
knn_precision = precision_score(y_test_class, knn_y_pred_class,zero_division=0)
knn_specificity = confusion_matrix(y_test_class, knn_y_pred_class)[0, 0] / (confusion_matrix(y_test_class, knn_y_pred_class)[0, 0] + confusion_matrix(y_test_class, knn_y_pred_class)[0, 1])
knn_recall = recall_score(y_test_class, knn_y_pred_class,zero_division=0)
knn_f1_score = f1_score(y_test_class, knn_y_pred_class,zero_division=0)

rf_accuracy = accuracy_score(y_test_class, rf_y_pred_class)
rf_error_rate = 1 - rf_accuracy
rf_precision = precision_score(y_test_class, rf_y_pred_class,zero_division=0)
rf_specificity = confusion_matrix(y_test_class, rf_y_pred_class)[0, 0] / (confusion_matrix(y_test_class, rf_y_pred_class)[0, 0] + confusion_matrix(y_test_class, rf_y_pred_class)[0, 1])
rf_recall = recall_score(y_test_class, rf_y_pred_class,zero_division=0)
rf_f1_score = f1_score(y_test_class, rf_y_pred_class,zero_division=0)

lr_accuracy = accuracy_score(y_test_class, lr_y_pred_class)
lr_error_rate = 1 - lr_accuracy
lr_precision = precision_score(y_test_class, lr_y_pred_class,zero_division=0)
lr_specificity = confusion_matrix(y_test_class, lr_y_pred_class)[0, 0] / (confusion_matrix(y_test_class, lr_y_pred_class)[0, 0] + confusion_matrix(y_test_class, lr_y_pred_class)[0, 1])
lr_recall = recall_score(y_test_class, lr_y_pred_class,zero_division=0)
lr_f1_score = f1_score(y_test_class, lr_y_pred_class,zero_division=0)

# Print classification metrics
print('Classification Metrics for K-NN:')
print('Accuracy:', round(knn_accuracy,2))
print('Error Rate:', round(knn_error_rate,2))
print('Precision:', round(knn_precision,2))
print('Specificity:', round(knn_specificity,2))
print('Recall:', round(knn_recall,2))
print('F1 Score:', round(knn_f1_score,2))

print('Classification Metrics for Random Forest:')
print('Accuracy:', round(rf_accuracy,2))
print('Error Rate:', round(rf_error_rate,2))
print('Precision:', round(rf_precision,2))
print('Specificity:', round(rf_specificity,2))
print('Recall:', round(rf_recall,2))
print('F1 Score:', round(rf_f1_score,2))

print('Classification Metrics for Linear Regression:')
print('Accuracy:', round(lr_accuracy,2))
print('Error Rate:', round(lr_error_rate,2))
print('Precision:', round(lr_precision,2))
print('Specificity:', round(lr_specificity,2))
print('Recall:', round(lr_recall,2))
print('F1 Score:', round(lr_f1_score,2))

# Plot confusion matrix for each model
plt.figure(figsize=(10, 8))
plt.subplot(5, 5, 1)
sns.heatmap(confusion_matrix(y_test_class, knn_y_pred_class), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (K-NN)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.subplot(5, 5, 2)
sns.heatmap(confusion_matrix(y_test_class, rf_y_pred_class), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.subplot(5, 5, 3)
sns.heatmap(confusion_matrix(y_test_class, lr_y_pred_class), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Linear Regression)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.tight_layout()
plt.show()

# Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values for each model
knn_fpr, knn_tpr, _ = roc_curve(y_test_class, knn_y_pred_class)
rf_fpr, rf_tpr, _ = roc_curve(y_test_class, rf_y_pred_class)
lr_fpr, lr_tpr, _ = roc_curve(y_test_class, lr_y_pred_class)

# Calculate the area under the ROC curve (AUC) for each model
knn_auc = auc(knn_fpr, knn_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
lr_auc = auc(lr_fpr, lr_tpr)

# Plot the ROC curves for all models
plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, color='blue', lw=2, label='K-NN (AUC = %0.2f)' % knn_auc)
plt.plot(rf_fpr, rf_tpr, color='red', lw=2, label='Random Forest (AUC = %0.2f)' % rf_auc)
plt.plot(lr_fpr, lr_tpr, color='green', lw=2, label='Linear Regression (AUC = %0.2f)' % lr_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
