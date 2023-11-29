import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import pickle

# Cargamos el dataframe procesado
df= pd.read_csv('../data/processed.csv')

#Separamos el dataframe en train y test y lo guardamos en .csv
train, test = train_test_split(df, test_size=0.2, random_state=42)
# guardamos los dos nuevos dataframes en .csv para usarlos mas tarde
train.to_csv('../data/trainnew.csv', index=False)
test.to_csv('../data/testnew.csv', index=False)

with open("trained_model_reg_2GGR.pkl", "rb") as archivo:
    modelo_final = pickle.load(archivo)

X = train.drop(columns=['quality'])  
y = train['quality']  

oversampler = RandomOverSampler(random_state=42)

X_resampled, y_resampled = oversampler.fit_resample(X, y)
df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
X=df_balanced.drop(columns=['quality'])
y=df_balanced['quality']

modelo_final.fit(X, y)

with open('../models/trained_model_reg_2GBRNEW.pkl', 'wb') as archivo_salida :
    pickle.dump(modelo_final, archivo_salida)