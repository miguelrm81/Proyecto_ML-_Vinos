import streamlit as st
import pickle
import pandas as pd

# Carga tus modelos aquí (reemplaza las rutas con las correctas)
with open('../models/trained_model_reg_2GBR.pkl', 'rb') as gbr:
    gbr_model = pickle.load(gbr)
with open('../models/trained_model_reg_1DTR.pkl', 'rb') as dt :
    dt_model = pickle.load(dt)

with open('../models/trained_model_reg_3RFR.pkl', 'rb') as rf :
    rf_model = pickle.load(rf)

with open('../models/trained_model_reg_4LASSO.pkl', 'rb') as lasso :
    lasso_model = pickle.load(lasso)

with open('../models/trained_model_reg_5SVC.pkl', 'rb') as svc :
    svc_model = pickle.load(svc)

with open('../models/trained_model_reg_6PCA.pkl', 'rb') as pca :
    pca_model = pickle.load(pca)

with open('../models/trained_model_reg_7PCAGB.pkl', 'rb') as pcagb :
    pcagb_model = pickle.load(pcagb)

with open('../models/trained_model_1rf.pkl', 'rb') as rfcla :
    rfcla_model = pickle.load(rfcla)

with open('../models/trained_model_2gb.pkl', 'rb') as gbcla :
    gbcla_model = pickle.load(gbcla)

with open('../models/trained_model_3lg.pkl', 'rb') as lgcla :
    lgcla_model = pickle.load(lgcla)

with open('../models/trained_model_4SVC.pkl', 'rb') as svcla :
    svcla_model = pickle.load(svcla)

with open('../models/trained_model_5KNN.pkl', 'rb') as knncla :
    knncla_model = pickle.load(knncla)

with open('../models/trained_model_6PCA.pkl', 'rb') as pcacla :
    pcacla_model = pickle.load(pcacla)


def add_logo():
    main_bg = "../docs/copas.jpg"
    st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url({main_bg}) no-repeat center center fixed;
            background-size: cover;
            background-color: #6f0000;  /* Rojo vino tinto */
        }}
    </style>
    """,
    unsafe_allow_html=True
)

def show_procesamiento():
     
    st.sidebar.title("Menú")
    submenu = st.sidebar.radio("Selecciona un submenú:", ["Gráficas", "Código"])

    
    st.title("Página de Procesamiento de Datos")
    if submenu == "Gráficas":
        st.image("../docs/fotos_vino/terget.jpg", caption="Variación del target")
        st.image("../docs/fotos_vino/matriz.jpg", caption="Variacion del heatmap")
        st.image("../docs/fotos_vino/boxplot_cali_alc.png", caption="Correlacion calidad - Alcohol")
        st.image("../docs/fotos_vino/histograma.png", caption="Histograma variables")
    elif submenu == "Código":
            st.text("Limpieza de outliers")
            st.code("""
                    df = df[df['total sulfur dioxide'] < 300] 
                    df = df[df['free sulfur dioxide'] < 120] 
                    df = df[df['residual sugar'] < 50]
                    """)
            st.text("Creacion de nuevas variables")
            st.code("""
                    df['type_num'] = df['type'].map({'white': 1, 'red': 2})
                    
                    # Calculamos la media de 'alcohol' asociada a cada valor de 'residual sugar'
                    med_alc_sug = df.groupby('residual sugar')['alcohol'].mean()
                    df['alc-sug'] = df['residual sugar'].map(med_alc_sug)
                    
                    # Calculamos la media de 'Quality' asociada a cada valor de 'residual sugar'
                    med_qua_sug = df.groupby('residual sugar')['quality'].mean()
                    df['qua-sug'] = df['residual sugar'].map(med_qua_sug)
                    
                    # Calculamos la media de 'volatile acidity' asociada a cada valor de 'type_num'
                    med_chlo_aci= df.groupby('type_num')['volatile acidity'].mean()
                    df['aci-type'] = df['type_num'].map(med_qua_sug)
                    """)
    
    

def show_entrenamiento():
    
    st.write("Página de Entrenamiento")
    st.sidebar.title("Menú")
    submenu = st.sidebar.radio("Selecciona un submenú:", ["Gráficas", "Código"])

    st.title("Página de Procesamiento de Datos")
    if submenu == "Gráficas":
        st.image("../docs/fotos_vino/1dtr.png", caption="Resultados Decission Tree")
        st.image("../docs/fotos_vino/2gbr.png", caption="Resultados Gradient Boosting")
        st.image("../docs/fotos_vino/3RFR.png", caption="Resultados Random Forest")
        st.image("../docs/fotos_vino/4LASSO.png", caption="Resultados LASSO")
        st.image("../docs/fotos_vino/5SVC.png", caption="Resultados SVC")
        st.image("../docs/fotos_vino/6PCA.png", caption="Resultados PCA + Random forest")
        st.image("../docs/fotos_vino/7PCAGBGRAF.png", caption="Resultados PCA + Gradient Boosting")
    elif submenu == "Código":
            st.text("Equilibrado del target")
            st.code("""
                    train = pd.read_csv('../data/train.csv')
                    X = train.drop(columns=['quality'])  
                    y = train['quality']  
                    oversampler = RandomOverSampler(random_state=42)
                    X_resampled, y_resampled = oversampler.fit_resample(X, y)
                    df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
                    """)
            st.text("Busqueda de modelo e hiperparametrización")
            st.code("""
                    pipe2 = Pipeline([
                    ('scaler', StandardScaler()),  
                    ('gb', GradientBoostingRegressor())  
                    ])
                    param2 = {
                        'gb__n_estimators': [50, 100, 200],
                        'gb__learning_rate': [ 0.2, 0.5, 1],
                        'gb__max_depth': [3, 4, 5,6]
                    }
                    gb_gs = GridSearchCV(pipe2, param2, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
                    gb_gs.fit(X_train, y_train)
                    best_model2 = gb_gs.best_estimator_
                    """)
            st.text("Obtención de resultados")
            st.code("""
                    mae = mean_absolute_error(y_test, y_pred2)
                    mape = np.mean(np.abs((y_test - y_pred2) / y_test)) * 100
                    mse = mean_squared_error(y_test, y_pred2)
                    rmse = np.sqrt(mse)

                    print(f'Mean Absolute Error (MAE): {mae}')
                    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
                    print(f'Mean Squared Error (MSE): {mse}')
                    print(f'Root Mean Squared Error (RMSE): {rmse}')
                    """)
            st.image("../docs/fotos_vino/2rgbMAE2.png", caption="Resultados Gradient Boosting")
def show_comprobador():
        
        st.write("Página de Comprobador")

        st.sidebar.header('User input parameters')

        def user_input_parameters():
            fixed_acidity = st.sidebar.slider('fixed acidity', 3.8, 15.9, 7.0)
            volatile_acidity = st.sidebar.slider('volatile acidity', 0.08, 1.58, 0.29)
            citric_acid = st.sidebar.slider('citric acid', 0.0, 1.66, 0.31)
            residual_sugar = st.sidebar.slider('residual sugar', 0.6, 31.6, 3.0)
            chlorides = st.sidebar.slider('chlorides', 0.01, 0.61, 0.05)
            free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1.0, 118.5, 29.0)
            total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6, 294, 118)
            density = st.sidebar.slider('density', 0.98, 1.01, 0.99)
            pH = st.sidebar.slider('pH', 2.72, 4.01, 3.21)
            sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 0.51)
            alcohol = st.sidebar.slider('alcohol', 8.0, 14.9, 10.3)
            type_num = st.sidebar.selectbox('type_num', [1, 2])
            alc_sug	 = st.sidebar.slider('alc-sug', 8.5, 13.4, 10.67)
            qua_sug	 = st.sidebar.slider('qua-sug', 4.0, 8.0, 5.81)
            aci_type = st.sidebar.slider('aci-type', 5.698, 5.71, 5.70)
            alc_citr = st.sidebar.slider('alc-citr', 0.12, 0.66, 0.48)
    
            data = {'fixed acidity' : fixed_acidity, 'volatile acidity' : volatile_acidity, 'citric acid' : citric_acid,
                    'residual sugar' : residual_sugar , 'chlorides' : chlorides,'free sulfur dioxide' : free_sulfur_dioxide, 
                    'total sulfur dioxide' : total_sulfur_dioxide, 'density' : density, 'pH' : pH, 'sulphates':sulphates, 'alcohol' : alcohol, 
                    'type_num' : type_num ,'alc-sug': alc_sug,'qua-sug': qua_sug,'aci-type': aci_type, 'alc-citr': alc_citr   }
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_parameters()

            # Selección de clasificación o regresión
        tipo_problema = st.radio("Selecciona el tipo de problema:", ["Clasificación", "Regresión"])

        if tipo_problema == "Clasificación":
            # Opciones de modelos para clasificación
            opcion_clasificacion = ['Random Forest', 'Gradiant Boosting', 'Logistic Regression', 'SVC', 'KNN', 'PCA + GB']
            model = st.selectbox('Seleccione el modelo de clasificación:', opcion_clasificacion)
        else:
            # Opciones de modelos para regresión
            opcion_regresion = ['Decission Tree Regression', 'Gradiant Boosting Regression', 'Random Forest Regression', 'Lasso', 'SVC', 'PCA + RF', 'PCA + GB']
            model = st.selectbox('Seleccione el modelo de regresión:', opcion_regresion)

        st.subheader('Parámetros seleccionados')
        st.subheader(model)
        st.write(df)

        if st.button('RUN'):
            if tipo_problema == "Clasificación":
                
                if model == 'Random Forest':
                    resultado = rfcla_model.predict(df)
                elif model == 'Gradiant Boosting':
                    resultado=gbcla_model.predict(df)
                elif model == 'Logistic Regression':
                    resultado=lgcla_model.predict(df)
                elif model == 'SVC':
                    resultado=svcla_model.predict(df)
                elif model == 'KNN':
                    resultado=knncla_model.predict(df)
                elif model == 'PCA + GB':
                    resultado=pcacla_model.predict(df)
            else:
                
                if model == 'Gradiant Boosting Regression':
                    resultado=gbr_model.predict(df)
                elif model == 'Decission Tree Regression':
                    resultado=dt_model.predict(df)
                elif model == 'Random Forest Regression':
                    resultado=rf_model.predict(df)
                elif model == 'Lasso':
                    resultado=lasso_model.predict(df)
                elif model == 'SVC':
                    resultado=svc_model.predict(df)
                elif model == 'PCA + RF':
                    resultado=pca_model.predict(df)
                elif model == 'PCA + GB':
                    resultado=pcagb_model.predict(df)
            st.success(f"Resultado de la predicción: {resultado}")

            if resultado < 5:  # Puedes ajustar esta condición según tu lógica
                st.image("../docs/brik.jpg")
            elif resultado >= 5 and resultado < 7 :
                st.image("../docs/medio.jpg")
            else :
                st.image("../docs/premium.jpg")



def main():
    add_logo()

    st.title('Modelo de Regresión Proyecto Vino Calidad')

    menu_option = st.sidebar.selectbox("Menú", ["Inicio", "Procesamiento", "Entrenamiento", "Comprobador"])

    if menu_option == "Inicio":
        # Muestra la imagen de inicio
        st.image("../docs/copas.jpg", use_column_width=True)
    elif menu_option == "Procesamiento":
        show_procesamiento()
    elif menu_option == "Entrenamiento":
        show_entrenamiento()
    elif menu_option == "Comprobador":
        show_comprobador()

if __name__ == "__main__":
    main()