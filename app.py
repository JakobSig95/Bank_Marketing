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

# Visualizing how Maritial Status and Education is distributed in the dataset.

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))



#______________________________________________________________________________________________________________________________


## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank Marketing 💸')

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Supervised Machine Learning", "Unsupervised Machine Learning"])

with tab1:

    st.text('Here is a chart showing the age distribution which we can see is right-skewed') 
    st.text('but besides from the outliers have a fairly normal distribution from the ages')
    st.text('of 20-61')

    import matplotlib.pyplot as plt
    
    import numpy as np
    fig, ax = plt.subplots()
    ax.hist(data['age'], bins=20)
    st.pyplot(fig)

    # First plot for marital status

    fig, axx = plt.subplots()
    axx.hist(data['marital'], bins=20)
    st.pyplot(fig)

    fig, axx = plt.subplots()
    axx.hist(data['education'], bins=20)
    st.pyplot(fig)

    import plotly.express as px
    mlabels=['basic 4y', 'high school','basic 6y','basic 9y','profesional course','unknown','university degree']
    educational_fig = px.pie(data, names=mlabels,values='size',hole = 0.8)
    educational_fig.update_traces(textposition='outside', textinfo='percent+label')
    educational_fig.update_layout(
    annotations=[dict(text="comparison of education", x=0.5, y=0.5, font_size=20, showarrow=False)])
    educational_fig.update_layout(showlegend=False)
    educational_fig.update_layout(height=500, width=600)

#sns.countplot(x = "marital", data = data, ax = ax1)
#ax1.set_title("marital status distribution", fontsize = 13)
#ax1.set_xlabel("Marital Status", fontsize = 12)
#ax1.set_ylabel("Count", fontsize = 12)

# Second plot for Education distribution

#sns.countplot(x = "education", data = data, ax = ax2)
#ax2.set_title("Education distribution", fontsize = 13)
#ax2.set_xlabel("Education level", fontsize = 12)
#ax2.set_ylabel("Count", fontsize = 12)
#ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 70)
    
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
