import streamlit as st
from xgboost import XGBRegressor
import pickle

## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank marketing predicting subscription')

# load the model from disk
loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)

#
