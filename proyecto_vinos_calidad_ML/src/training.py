import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler

# Cargamos el dataframe procesado
df= pd.read_csv('../data/processed.csv')

#Separamos el dataframe en train y test y lo guardamos en .csv
train, test = train_test_split(df, test_size=0.2, random_state=10)
# guardamos los dos nuevos dataframes en .csv para usarlos mas tarde
train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)



