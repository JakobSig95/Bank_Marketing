import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import ConnectionPatch
import plotly.express as px


## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank marketing predicting subscription')

st.title('Bank Marketing 💸')

tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Gender and fatality", "Attacks around the world", "Activities and year"])

# load the model from disk
loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)

#
