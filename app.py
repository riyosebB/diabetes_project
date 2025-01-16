import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_diabetes


# Streamlit app title
st.title('This app predicts the glucose level in the blood of a diabetic patient')

# Load models using `pickle`
with open(r'Models\model_lr.pkl', 'rb') as file_lr:
    model_lr = pickle.load(file_lr)

with open(r'Models\model_en.pkl', 'rb') as file_em:
    model_en = pickle.load(file_em)

with open(r'Models\model_ridge.pkl', 'rb') as file_ridge:
    model_ridge = pickle.load(file_ridge)

# Load the dataset
diab = load_diabetes()
X = pd.DataFrame(diab.data, columns=diab.feature_names)




# User input 
st.subheader('Input Features')
user_input = {}

for col in X.columns:
    user_input[col] = st.slider(col, X[col].min(),X[col].max(), X[col].mean())

# Create a DataFrame from user input
df = pd.DataFrame(user_input, index=[0])
st.write('User Input:', df)

models = {'Linear Regression':model_lr, 'Ridge Model': model_ridge, 'Elastic Net':model_en}



selected_models= st.selectbox('Select teh models',('Linear Regression', 'Ridge', 'Elastic Net'))

# Prediction
if st.button('Predict'):
    prediction = models[selected_models].predict(df)[0]
    st.write(f'The predicted glucose level is **{prediction:.2f}**')









