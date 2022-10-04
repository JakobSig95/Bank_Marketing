import streamlit as st

!pip install xgboost -U -q #Machine learning packages - Checking for previous versions, drops it and installs the newest
!pip install sklearn -U -q #Machine learning packages - Checking for previous versions, drops it and installs the newest

!pip install pydeck -q #Interactive data visualization - Checking for previous versions, drops it and installs the newest
!pip install folium #Geoplotting

import pandas as pd #Data analysis and processing tool
import numpy as np #Mathematical functions
import seaborn as sns #Seaborn plots
from matplotlib import pyplot as plt #Plot control
sns.set() #Plot style
import altair as alt #declarative statistical visualization library
from vega_datasets import data #declarative statistical visualization library
%matplotlib inline

#Geoplotting with folium/leaflet
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap

#Fancy geoplotting with DeckGL
import pydeck as pdk

from sklearn.preprocessing import LabelEncoder #Predictive data analysis
from imblearn.under_sampling import NearMiss #Class to perform under-sampling
from scipy import stats #Provides more utility functions for optimization, stats and signal processing

