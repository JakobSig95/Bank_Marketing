import streamlit as st
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd #Data analysis and processing tool
import numpy as np #Mathematical functions
import seaborn as sns #Seaborn plots
from matplotlib import pyplot as plt #Plot control
sns.set() #Plot style
import altair as alt #declarative statistical visualization library

from imblearn.under_sampling import NearMiss #Class to perform under-sampling
from scipy import stats #Provides more utility functions for optimization, stats and signal processing


data = pd.read_csv("https://github.com/JakobSig95/bank_Marketing/raw/main/bank_marketing.csv", delimiter=';')

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

    fig, ax = plt.subplots()
    ax.hist(data['job'], bins=20)
    st.pyplot(fig)



#sns.cuntplot(x = "marital", data = data, ax = ax1)
#ax1.set_title("marital status distribution", fontsize = 13)
#ax1.set_xlabel("Marital Status", fontsize = 12)
#ax1.set_ylabel("Count", fontsize = 12)

# Second plot for Education distribution

#sns.countplot(x = "education", data = data, ax = ax2)
#ax2.set_title("Education distribution", fontsize = 13)
#ax2.set_xlabel("Education level", fontsize = 12)
#ax2.set_ylabel("Count", fontsize = 12)
#ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 70)
    
#with st.sidebar:
    
 #   "Made by:"
  #  "Mikkel"
   # "Jakob"
    #"Alpha"


# ["age", "duration", "emp.var.rate", "job", "euribor3m", "nr.employed"]
    
model_xgb = pickle.load(open('model_xgb.pkl','rb'))
res = model_xgb.predict(np.array([56, 261, 1.1, 1, 4.857, 5191.0]))
st.write(res)
