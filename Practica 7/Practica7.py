import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns
###
url = "https://raw.githubusercontent.com/cesar3245/Mineria-de-datos/main/Practica%201/dataset/Police%20Data.csv"
ds = pd.read_csv(url)
###

###
ds = ds.drop(columns=['country_name', 'search_type','driver_age_raw'])

ds = ds.dropna(axis=0, how='any')

ds = ds[ds['stop_duration'] != '2']
###

###
ds['stop_date'] = pd.to_datetime(ds['stop_date'], format='%m/%d/%Y', errors='coerce')

ds['stop_year'] = ds['stop_date'].dt.year
ds['stop_month'] = ds['stop_date'].dt.month
ds['stop_day'] = ds['stop_date'].dt.day
ds['stop_wday'] = ds['stop_date'].dt.day_name()
###

###
stop_time2 = pd.to_datetime(ds['stop_time'], format='%H:%M', errors='coerce')

ds['stop_hour'] = stop_time2.dt.hour
ds['stop_minute'] = stop_time2.dt.minute
###

###
ds = ds.dropna(subset=['stop_year', 'stop_month', 'stop_day', 'stop_wday', 'stop_hour', 'stop_minute'])
###

###
ds2 = ds.drop(columns=['stop_date','stop_time'])
###

###
ds3 = ds2.copy()

ds3['search_conducted'] = ds3['search_conducted'].astype(int)
ds3['drugs_related_stop'] = ds3['drugs_related_stop'].astype(int)
ds3['driver_gender'] = ds3['driver_gender'].map({'M': 1, 'F': 0})
ds3['is_arrested'] = ds3['is_arrested'].map({'True': 1, 'False': 0, True: 1, False: 0})
###

###
ds3_cat = ds3.select_dtypes(include='object').columns.tolist()

cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

ds3_cat_tr = cat_encoder.fit_transform(ds3[ds3_cat])

ds3_cat_df = pd.DataFrame(ds3_cat_tr, columns=cat_encoder.get_feature_names_out(ds3_cat), index=ds3.index)

ds4 = pd.concat([ds3.drop(columns=ds3_cat), ds3_cat_df], axis=1)
###

###
ds5 = ds4.copy()  
###

###
X = ds5.drop(columns=['is_arrested', 'stop_outcome_Arrest Driver'])   
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=ds5.index)
###

###
clusters = range(2, 11)
inercias = []
for i in clusters:
    mods = KMeans(n_clusters=i, random_state=42, n_init=10).fit(X)
    inercias.append(mods.inertia_)

plt.plot(list(clusters), inercias, marker='o')
plt.xlabel('k'); plt.ylabel('Inercia'); plt.title('Metodo del codo')
plt.tight_layout(); plt.show()
###

###
k = 5
modelo3 = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
ds5['cluster'] = modelo3.labels_
###

###
print(f"Silhouette (k={k}): {silhouette_score(X, modelo3.labels_):.3f}")
columnas = ['is_arrested', 'drugs_related_stop', 'violation_raw_Speeding', 'driver_age', 'stop_hour']
print("\nPromedios por cluster:")
print(ds5.groupby('cluster')[columnas].mean().round(3))
###

###
m = ds5.sample(n=6000, random_state=42)
sns.scatterplot(data=m, x='stop_hour', y='driver_age', hue='cluster', palette='viridis', s=12, linewidth=0)
#sns.scatterplot(data=ds5, x='stop_hour', y='driver_age', hue='cluster',palette='viridis', s=5, linewidth=0, alpha=0.5)
plt.title(f'Clusters por hora y edad (k={k})')
plt.tight_layout(); plt.show()
###

###
ds5.groupby('cluster')['is_arrested'].mean().plot(kind='bar', rot=0)
plt.title('Tasa de arresto por cluster'); plt.ylabel('Promedio')
plt.tight_layout(); plt.show()
###

###
print("\nCentros de los clusters:")
print(pd.DataFrame(modelo3.cluster_centers_, columns=X.columns).iloc[:, :5])
###
