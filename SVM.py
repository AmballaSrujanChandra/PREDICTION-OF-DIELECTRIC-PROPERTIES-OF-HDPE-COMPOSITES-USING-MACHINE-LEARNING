# Predicting Dielectric loss (D) and Dielectric constant (k) of HDPE TiO2 using Support Vector Machine (SVM) Regressor .Here, Frequency and HDPE% is feature variables and  D and K is target variables.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and merge data
data_files = ["H0T.xlsx", "H10T.xlsx", "H20T.xlsx", "H30T.xlsx", "H40T.xlsx"]
dfs = [pd.read_excel(file) for file in data_files]
# Concatenate dataframes
data = pd.concat(dfs, ignore_index=True)
# Convert the 'D' column to numeric, treating non-numeric values as NaN
data['D'] = pd.to_numeric(data['D'], errors='coerce')

# Fill missing values with the mean
data['D'].fillna(data['D'].mean(), inplace=True)

# Remove the 'S.no' and 'Cp (1.5 mm)' columns
columns_to_remove = ['S.no', 'Cp (1.5mm)']
data.drop(columns=columns_to_remove, inplace=True)

# Load and merge data
data_files = ["H0T.xlsx", "H10T.xlsx", "H20T.xlsx", "H30T.xlsx", "H40T.xlsx"]
dfs = [pd.read_excel(file) for file in data_files]
# Concatenate dataframes
data = pd.concat(dfs, ignore_index=True)

# Convert the 'D' column to numeric, treating non-numeric values as NaN
data['D'] = pd.to_numeric(data['D'], errors='coerce')

# Fill missing values with the mean
data['D'].fillna(data['D'].mean(), inplace=True)

# Remove the 'S.no' and 'Cp (1.5 mm)' columns
columns_to_remove = ['S.no', 'Cp (1.5mm)']
data.drop(columns=columns_to_remove, inplace=True)

print(data)

# Split data into features (X) and targets (y)
X = data[['Fre (Hz)', 'HDPE%']]
y = data[['D', 'K']]  # Separate D and K

# Split data into training and testing sets for both 'D' and 'K'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM Regressor model for 'D'
svm_model_D = SVR(kernel='linear')
svm_model_D.fit(X_train, y_train['D'])  # Train for 'D'

# Train an SVM Regressor model for 'K'
svm_model_K = SVR(kernel='linear')
svm_model_K.fit(X_train, y_train['K'])  # Train for 'K'

# Evaluate the model for 'D'
y_pred_D = svm_model_D.predict(X_test)
mse_D = mean_squared_error(y_test['D'], y_pred_D)
r2_D = r2_score(y_test['D'], y_pred_D)
print(f"Mean Squared Error (MSE) for 'D': {mse_D}")
print(f"R-squared (R2) for 'D': {r2_D}")

# Evaluate the model for 'K'
y_pred_K = svm_model_K.predict(X_test)
mse_K = mean_squared_error(y_test['K'], y_pred_K)
r2_K = r2_score(y_test['K'], y_pred_K)
print(f"Mean Squared Error (MSE) for 'K': {mse_K}")
print(f"R-squared (R2) for 'K': {r2_K}")

# Show the relationship between Fre(Hz), HDPE%, D, and K using a pairplot
data['D Predicted'] = svm_model_D.predict(X)  # Predictions for 'D'
data['K Predicted'] = svm_model_K.predict(X)  # Predictions for 'K'

sns.pairplot(data[['Fre (Hz)', 'HDPE%', 'D', 'K', 'D Predicted', 'K Predicted']])
plt.show()

# Scatter plot for 'Fre (Hz)' vs. 'D'
plt.figure(figsize=(12, 5))
# Actual 'D' values
plt.scatter(data['Fre (Hz)'], data['D'], label='Actual D', alpha=0.7, c='blue', marker='o', s=50)
# Predicted 'D' values
plt.scatter(data['Fre (Hz)'], data['D Predicted'], label='Predicted D', alpha=0.7, c='red', marker='x', s=50)
plt.xlabel('Fre (Hz)')
plt.ylabel('D')
plt.title('Actual vs. Predicted D (Dielectric Loss)')
plt.legend()

# Scatter plot for 'HDPE%' vs. 'D'
plt.figure(figsize=(12, 5))
# Actual 'D' values
plt.scatter(data['HDPE%'], data['D'], label='Actual D', alpha=0.7, c='blue', marker='o', s=50)
# Predicted 'D' values
plt.scatter(data['HDPE%'], data['D Predicted'], label='Predicted D', alpha=0.7, c='red', marker='x', s=50)
plt.xlabel('HDPE%')
plt.ylabel('D')
plt.title('Actual vs. Predicted D (Dielectric Loss)')
plt.legend()
plt.show()

# Scatter plot for 'Fre (Hz)' vs. 'K'
plt.figure(figsize=(12, 5))
# Actual 'K' values
plt.scatter(data['Fre (Hz)'], data['K'], label='Actual K', alpha=0.7, c='blue', marker='o', s=50)
# Predicted 'K' values
plt.scatter(data['Fre (Hz)'], data['K Predicted'], label='Predicted K', alpha=0.7, c='red', marker='x', s=50)
plt.xlabel('Fre (Hz)')
plt.ylabel('K')
plt.title('Actual vs. Predicted K (Dielectric Constant)')
plt.legend()

# Scatter plot for 'HDPE%' vs. 'K'
plt.figure(figsize=(12, 5))
# Actual 'K' values
plt.scatter(data['HDPE%'], data['K'], label='Actual K', alpha=0.7, c='blue', marker='o', s=50)
# Predicted 'K' values
plt.scatter(data['HDPE%'], data['K Predicted'], label='Predicted K', alpha=0.7, c='red', marker='x', s=50)
plt.xlabel('HDPE%')
plt.ylabel('K')
plt.title('Actual vs. Predicted K (Dielectric Constant)')
plt.legend()
plt.show()

# Determine how reliable the combined model is - Cross-validation
from sklearn.model_selection import cross_val_score
cross_val_scores_D = cross_val_score(svm_model_D, X, y['D'], cv=5, scoring='neg_mean_squared_error')
mean_cv_mse_D = -np.mean(cross_val_scores_D)
print(f"Mean Cross-Validation MSE for 'D' model: {mean_cv_mse_D}")

cross_val_scores_K = cross_val_score(svm_model_K, X, y['K'], cv=5, scoring='neg_mean_squared_error')
mean_cv_mse_K = -np.mean(cross_val_scores_K)
print(f"Mean Cross-Validation MSE for 'K' model: {mean_cv_mse_K}")

# User input and prediction
while True:
    try:
        frequency = input("Enter Frequency (Fre (Hz)): ")
        
        # Check if the user wants to exit
        if frequency.lower() == 'exit':
            print("Exiting...")
            break
        
        hdpe_percentage = input("Enter HDPE Percentage: ")

        # Check if the user wants to exit
        if hdpe_percentage.lower() == 'exit':
            print("Exiting...")
            break

        # Attempt to convert input to float
        try:
            frequency = float(frequency)
            hdpe_percentage = float(hdpe_percentage)
        except ValueError:
            raise ValueError("Invalid input. Please enter valid numbers.")

        # Create a dataframe with user input
        user_data = pd.DataFrame({'Fre (Hz)': [frequency], 'HDPE%': [hdpe_percentage]})

        # Predict 'D' and 'K' separately
        prediction_D = svm_model_D.predict(user_data)
        prediction_K = svm_model_K.predict(user_data)

        print(f"Predicted D (Dielectric Loss): {prediction_D[0]}")
        print(f"Predicted K (Dielectric Constant): {prediction_K[0]}")
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("\nExiting...")
        break