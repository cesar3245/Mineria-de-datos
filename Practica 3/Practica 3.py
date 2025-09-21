import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
plt.figure()
ds5['driver_age'].dropna().plot(kind='hist', bins=30, edgecolor='black')
plt.title('Edades de conductores')
plt.xlabel('Edad')
plt.ylabel('Conductores')
plt.show()
###

###
det_hora = ds5.groupby('stop_hour').size()
plt.figure()
det_hora.plot(kind='bar')
plt.title('Detenciones por hora')
plt.xlabel('Hora')
plt.ylabel('Detenciones')
plt.show()
###

###
cols_race = [i for i in ds5.columns if i.startswith('driver_race_')]
arrested = ds5[ds5['stop_outcome_Arrest Driver'] == 1]
arrest_n = arrested[cols_race].sum().sort_values(ascending=False)
plt.figure()
plt.pie(arrest_n.values, labels=[x.replace('driver_race_', '') for x in arrest_n.index], autopct='%1.1f%%', startangle=90)
plt.title('Conductores arrestados por raza')
plt.show()
###

###
arrested_e = ds5.loc[ds5['stop_outcome_Arrest Driver'] == 1, 'driver_age']
plt.figure()
arrested_e.plot(kind='hist', bins=30, edgecolor='black')
plt.title('Edades de conductores arrestados')
plt.xlabel('Edad')
plt.ylabel('Conductores')
plt.show()
###

###
plt.figure()
ds5.boxplot(column='driver_age', by='stop_hour', grid=False)
plt.title('Edad de conductores por hora de detencion')
plt.suptitle('') 
plt.xlabel('Hora')
plt.ylabel('Edad')
plt.show()
###