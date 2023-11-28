import streamlit as st
import pickle
import pandas as pd

with open('../models/trained_model_reg_2GBR.pkl', 'rb') as gbr :
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



def main():
    st.title('Modelo de Regresion Proyecto Vino Calidad')

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

    opcion = ['Decission Tree Regression', 'Gradiant Boosting Regression', 'Random Forest Regression', 'Lasso', 'SVC', 'PCA + RF', 'PCA + GB']
    model= st.sidebar.selectbox('Que modelo quieres practicar?', opcion)
    
    st.subheader('Parametros seleccionados')
    st.subheader(model)
    st.write(df)
    if st.button('RUN'):
        if model == 'Gradiant Boosting Regression':
            st.success(gbr_model.predict(df))
        elif model == 'Decission Tree Regression' : 
            st.success((dt_model.predict(df)))
        elif model == 'Random Forest Regression' : 
            st.success((rf_model.predict(df)))
        elif model == 'Lasso' : 
            st.success((lasso_model.predict(df)))
        elif model == 'SVC' : 
            st.success((svc_model.predict(df)))
        elif model == 'PCA + RF' : 
            st.success((pca_model.predict(df)))
        elif model == 'PCA + GB' : 
            st.success((pcagb_model.predict(df)))

if __name__ == '__main__':
    main()