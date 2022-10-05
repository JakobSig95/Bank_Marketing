import streamlit as st
import pickle
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="💸")

st.title('Bank marketing predicting subscription')

# load the model from disk
loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)


