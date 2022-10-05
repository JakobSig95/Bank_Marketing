import streamlit as st
from xgboost import XGBRegressor
import pickle
pickle.dump(scaler, open('scaler.pkl','wb'))


## Importing packages and data
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
data = pd.read_csv("https://github.com/JakobSig95/Bank_Marketing/raw/main/bank_marketing.csv", delimiter=';')
## Getting an overview of the data
# Converting categorical into boolean using get_dummies 
# Getting the predicted values in terms of 0 and 1

Y = (data['y'] == 'yes')*1
#Getting an overview of the data set/data types

data.info()
##Getting an overview of the data set/data types

data.head()
data['y'].value_counts()
#Getting an overview of the data set/data types

data.tail()
#Getting an overview of the data set/data types

data.columns
# Dropping y from the original dataset as we have read it seperately

# data.drop('y', axis = 1, inplace = True)
# First five rows of the dataset after dropping y from the dataset

print(data.head())
data.describe()
## Exploratory Data Analysis
# Visaulizing how age is distributed in the dataset

sns.distplot(data['age'], hist = True, color = "#EE3B3B", hist_kws = {'edgecolor':'black'})
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = data, orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)
# Visualizing how Maritial Status and Education is distributed in the dataset.

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

# First plot for marital status

sns.countplot(x = "marital", data = data, ax = ax1)
ax1.set_title("marital status distribution", fontsize = 13)
ax1.set_xlabel("Marital Status", fontsize = 12)
ax1.set_ylabel("Count", fontsize = 12)

# Second plot for Education distribution

sns.countplot(x = "education", data = data, ax = ax2)
ax2.set_title("Education distribution", fontsize = 13)
ax2.set_xlabel("Education level", fontsize = 12)
ax2.set_ylabel("Count", fontsize = 12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 70)
#Visualizing how Jobs are distribution

fig, ax = plt.subplots()
fig.set_size_inches(15,5)
sns.countplot(x = "job", data = data)
ax.set_xlabel('Job', fontsize = 12)
ax.set_ylabel('Count', fontsize = 12)
ax.set_title("Job Count Distribution", fontsize = 13)
# Housing loan data distribution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.countplot(x = "housing", data = data, ax = ax1, order = ['yes', 'no', 'unknown'])
ax1.set_title("Housing Loan distribution")
ax1.set_xlabel("Housing Loan")
ax1.set_ylabel("Count")

# Personal loan data distribution
sns.countplot(x = "loan", data = data, ax = ax2, order = ['yes', 'no', 'unknown'])
ax2.set_title("Personal Loan Distribution")
ax2.set_xlabel("Personal Loan")
ax2.set_ylabel("Count")
Getting total count for:

Credit Defaulters
People with Housing loan
People with Personal loan
#Credit defaulter

print("Number of people with credit default: ", data[data['default'] == 'yes']['default'].count())
print("Number of people with no credit default: ", data[data['default'] == 'no']['default'].count())
print("Number of people who's credit default is unknown: ", data[data['default'] == 'unknown']['default'].count())
#Housing loan

print("Number of people with Housing loan: ", data[data['housing'] == 'yes']['housing'].count())
print("Number of people with no Housing loan: ", data[data['housing'] == 'no']['housing'].count())
print("Number of people who's Housing loan is unknown: ", data[data['housing'] == 'unknown']['housing'].count())
#Personal loan

print("Number of people with Personal loan: ", data[data['loan'] == 'yes']['loan'].count())
print("Number of people with no Personal loan: ", data[data['loan'] == 'no']['loan'].count())
print("Number of people who's Personal loan is unknown: ", data[data['loan'] == 'unknown']['loan'].count())
#Visualisation related to "Last Contact of the Current Campaign"
#Visualisation related to Duration
#Plotting duration using boxplot makes it difficult to obtain some important values like average of distribution
#and so I am plotting histogram on the side to see how its distributed and check for mean value (If its possible).

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))

sns.boxplot(x = "duration", data = data, orient = 'v', ax = ax1)
ax1.set_xlabel("Calls")
ax1.set_ylabel("Duration")
ax1.set_title("Call distribution")

sns.distplot(data['duration'], ax = ax2)
ax2.set_xlabel("Call duration")
ax2.set_ylabel("Count")
ax2.set_title("Call Duration vs Count")
#Evt. lav interaktivt kort over housing loans bla bla
data
print("Jobs: \n", data["job"].unique(),'\n')
print("Marital Status: \n", data['marital'].unique(),'\n')
print("Education: \n", data['education'].unique(),'\n')
print("Default on Credit: \n", data['default'].unique(),'\n')
print("Housing loan: \n", data['housing'].unique(),'\n')
print("Loan default: \n", data['loan'].unique(),'\n')
print("Contact type: \n", data['contact'].unique(),'\n')
print("Months: \n", data['month'].unique(),'\n')
print("day_of_week: \n", data['day_of_week'].unique(),'\n')
print("Poutcome: \n",data["poutcome"].unique(),'\n')
labelencoder_X = LabelEncoder()
data["job"] = labelencoder_X.fit_transform(data["job"])
data["marital"] = labelencoder_X.fit_transform(data["marital"])
data["education"] = labelencoder_X.fit_transform(data["education"])
data["default"] = labelencoder_X.fit_transform(data["default"])
data["housing"] = labelencoder_X.fit_transform(data["housing"])
data["loan"] = labelencoder_X.fit_transform(data["loan"])
data["contact"] = labelencoder_X.fit_transform(data["contact"])
data["month"] = labelencoder_X.fit_transform(data["month"])
data["day_of_week"] = labelencoder_X.fit_transform(data["day_of_week"])
data["poutcome"] = labelencoder_X.fit_transform(data["poutcome"])
data["y"] = labelencoder_X.fit_transform(data["y"])
pd.set_option('max_columns', None)
data.head()
data.y
data.y.value_counts()
undersample = NearMiss(version=3)
undersample
data
df_x = data.iloc[:,:-1]
df_y = data['y']
df_x.info()
df_y = labelencoder_X.fit_transform(df_y)
df_y
df_x
X, y = undersample.fit_resample(df_x, df_y)
X['y'] = y
X.head()
X['y'].value_counts()
fig, ax = plt.subplots(figsize = (20, 10))
matrix = np.triu(X.corr())
sns.heatmap(data.corr(), annot=True, fmt='.1f', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask=matrix)
## Machine Learning
X = data.iloc[:,:-1]
y = data.iloc[:,-1:]
data.y.value_counts()
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)
X_ros['y'] = y_ros
X_ros['y'].value_counts()
from sklearn.neighbors import KNeighborsClassifier
data_1 = X[["age", "duration", "emp.var.rate", "job", "euribor3m", "nr.employed"]]
knn = KNeighborsClassifier(n_neighbors=5) #making an classifier KNN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_1, y, random_state=12)
knn.fit(X_train, y_train)

knn.score(X_test, y_test) #how good the model predicts 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=42)

# lav evt et loop 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred) # søg confusion og se billede (flask positiv etc.)
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)
print(confusion_matrix(y_test, xgbprd ))
print(round(accuracy_score(y_test, xgbprd),2)*100)
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())

headers = ["name", "score"]
values = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (XGB)')

plt.show()
### uml
# unsupervised ML 
# We select only numerical features from the dataframe
# naming is in anticipation of future clustering
data_to_cluster = data.iloc[:,0:20]
# import and instantiate scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# learn x-y relationships (principal components) and transform
data_to_cluster_scaled = scaler.fit_transform(data_to_cluster)
# very similar syntax for min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler_min_max = MinMaxScaler()
data_to_cluster_minmax = scaler_min_max.fit_transform(data_to_cluster)
data_to_cluster
sns.displot(data=data_to_cluster, 
            x="duration",
            kind="kde")
sns.displot(data=pd.DataFrame(data_to_cluster_scaled, columns=data_to_cluster.columns), 
            x="duration",
            kind="kde")
# load up and instantiate PCS
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# fit-transform the data
data_reduced_pca = pca.fit_transform(data_to_cluster_scaled)
print(pca.components_)
pca.components_.shape
print(pca.explained_variance_ratio_)
sns.scatterplot(data_reduced_pca[:,0],data_reduced_pca[:,1])
vis_data = pd.DataFrame(data_reduced_pca)
vis_data['duration'] = data['duration']
vis_data['nr.employed'] = data['nr.employed']
vis_data.columns = ['x', 'y', 'duration', 'nr.employed']
#Måske undersampling?
plt.figure(figsize=(18,2))
sns.heatmap(pd.DataFrame(pca.components_, columns=data_to_cluster.columns), annot=True)
#quick correlation check

# Compute the correlation matrix
corr = data_to_cluster.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
data_to_cluster_scaled.iloc
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from sklearn.decomposition import NMF
!pip install umap-learn -q
import umap
# we totally could specify more than 2 dimensions (as well as some other parameters)
umap_scaler = umap.UMAP()
# umap accepts standard-scaled data
embeddings = umap_scaler.fit_transform(data_to_cluster_scaled)
# just as PCA, umap reduced data can be plottet
sns.scatterplot(embeddings[:,0],embeddings[:,1])
alt.data_transformers.enable('default', max_rows=None)
#alt.Chart(vis_data).mark_circle(size=60).encode(
#    x='x',
#    y='y',
#    tooltip=['duration', 'nr.employed']
#).interactive()
from sklearn.cluster import KMeans
clusterer = KMeans(n_clusters=5)
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_to_cluster_scaled)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
umap_scaler_km = umap.UMAP(n_components=2)
embeddings_km = umap_scaler.fit_transform(data_to_cluster_scaled)


Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(embeddings_km)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# back to our k-means instance. We take 3 clusters on non-reduced data
clusterer.fit(data_to_cluster_scaled)
# we can then copy the cluster-numbers into the original file and start exploring
data['cluster'] = clusterer.labels_
vis_data = pd.DataFrame(embeddings)
vis_data['emp.var.rate'] = data['emp.var.rate']
vis_data['cluster'] = data['cluster']
vis_data['cons.price.idx'] = data['cons.price.idx']
vis_data.columns = ['x', 'y', 'emp.var.rate', 'cluster','cons.price.idx']
vis_data
alt.Chart(vis_data).mark_circle(size=6).encode(
    x='x',
    y='y',
    tooltip=['emp.var.rate', 'cons.price.idx'],
    color=alt.Color('cluster:N', scale=alt.Scale(scheme='dark2')) #use N after the var to tell altair that it's categorical
).interactive()
### Extraction of the files


## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank marketing predicting subscription')


# load the model from disk
loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)


