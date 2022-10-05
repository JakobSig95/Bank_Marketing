import streamlit as st
import pandas as pd #Data analysis and processing tool
import numpy as np #Mathematical functions
import seaborn as sns #Seaborn plots
from matplotlib import pyplot as plt #Plot control
sns.set() #Plot style
import altair as alt #declarative statistical visualization library

from imblearn.under_sampling import NearMiss #Class to perform under-sampling
from scipy import stats #Provides more utility functions for optimization, stats and signal processing


data = pd.read_csv("https://github.com/JakobSig95/Bank_Marketing/raw/main/bank_marketing.csv", delimiter=';')

# Converting categorical into boolean using get_dummies 
# Getting the predicted values in terms of 0 and 1

Y = (data['y'] == 'yes')*1

#Getting an overview of the data set/data types

age_distribution = sns.distplot(data['age'], hist = True, color = "#EE3B3B", hist_kws = {'edgecolor':'black'})
#______________________________________________________________________________________________________________________________


## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank Marketing 💸')

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Supervised Machine Learning", "Unsupervised Machine Learning"])

with tab1:

    st.header('.')

    import matplotlib.pyplot as plt
    
    import numpy as np
    fig, ax = plt.subplots()
    ax.hist(data['age'], bins=20)
    st.pyplot(fig)
    
with st.sidebar:
    
    "Made by:"
    "Mikkel"
    "Jakob"
    "Alpha"
    
    
    

# load the model from disk
#loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)

#
