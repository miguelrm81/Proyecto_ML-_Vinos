import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('../data/test.csv')

with open('../models/trained_model_reg_2GBR.pkl', 'rb') as archivo:
    final_model = pickle.load(archivo)

X = df.drop(columns='quality')
y = df['quality']



y_pred = final_model.predict(X)

mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
