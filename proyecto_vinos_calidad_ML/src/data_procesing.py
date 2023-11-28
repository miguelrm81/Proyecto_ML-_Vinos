import pandas as pd

#cargamos el .csv
df = pd.read_csv('../data/raw/wine-quality-white-and-red.csv')

#creamos nuevas variables y limpiamos columnas
df['type_num'] = df['type'].map({'white': 1, 'red': 2})

med_alc_sug = df.groupby('residual sugar')['alcohol'].mean()
df['alc-sug'] = df['residual sugar'].map(med_alc_sug)

med_qua_sug = df.groupby('residual sugar')['quality'].mean()
df['qua-sug'] = df['residual sugar'].map(med_qua_sug)

med_chlo_aci= df.groupby('type_num')['volatile acidity'].mean()
df['aci-type'] = df['type_num'].map(med_qua_sug)

df['alc-citr'] = df['alcohol'] * df['chlorides']

df.drop(columns='type', inplace=True)

#eliminamos los outliers
df = df[df['total sulfur dioxide'] < 300]
df = df[df['free sulfur dioxide'] < 120]
df = df[df['residual sugar'] < 50]

#Guardamos el dataframe ya procesado
df.to_csv('../data/processed.csv', index=False)






