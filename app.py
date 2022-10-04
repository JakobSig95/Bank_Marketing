import streamlit as st
import pickle
from xgboost import XGBRegressor

st.write('Hello')

# load the model from disk
loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)

