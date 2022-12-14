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

    st.text('This chart shows the marital distribution, which shows most of the population')
    st.text('are maried in this case')

    fig, axx = plt.subplots()
    axx.hist(data['marital'], bins=20)
    st.pyplot(fig)

    st.text('This chart shows despite the blurry text the education distribution')
    st.text('which is differentiated between basic.4y, high.school, basic.6y')
    st.text('basic 9y, professional.course, unknown, university.degree and')
    st.text('illiterate. The chart shows that there are most people with,')
    st.text('a high school and a university degree.')

    fig, axx = plt.subplots()
    axx.hist(data['education'], bins=20)
    st.pyplot(fig)

    st.text('This chart shows the job count distribution and the categories are')
    st.text('housemaid, services, administrative, blue-collar, technician,')
    st.text('retired, management, unemployed, self-employed, unknown entrepeneur')
    st.text('and student. It shows that the majority are working in administrative,')
    st.text('blue-collar and technician')

    fig, ax = plt.subplots()
    ax.hist(data['job'], bins=20)
    st.pyplot(fig)

with tab2:

    st.text('On this page we would have liked to give the user the option to choose')
    st.text('between different variables and then the result would be shown in the')
    st.text('box where we currently only see three 0-numbers.')

    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    res = model_xgb.predict(np.array([[2.4000e+01, 1.3900e+02, 1.4000e+00, 7.0000e+00, 4.9620e+00,5.2281e+03]]))
    st.write(res)

with tab3:
    
    st.text('We have made and trained a model to learn general patterns in our dataset')
    st.text('and we would have represented the data in a more compressed way on this page')
    st.text('with identifying the related observations with clusters')
    st.text('it can be seen in the notebook though.')


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
    
with st.sidebar:
    
    "Made by:"
    "Mikkel"
    "Jakob"
    "Alpha"


# ["age", "duration", "emp.var.rate", "job", "euribor3m", "nr.employed"]
    

