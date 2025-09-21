import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, kruskal

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
f_a = ds5.loc[ds5['driver_gender']==0, 'driver_age'].dropna()  
m_a = ds5.loc[ds5['driver_gender']==1, 'driver_age'].dropna()  

t1, p1 = ttest_ind(f_a, m_a, equal_var=False)
print("T-test edad vs genero:", "t =", round(t1,3), "   p =", p1)
###

###
race_df = ds[['driver_race', 'driver_age']].dropna()
race_g = [a['driver_age'].values for _, a in race_df.groupby('driver_race')]

anova1, p2 = f_oneway(*race_g)
print("ANOVA edad vs raza: F =", round(anova1,3), "  p =", p2)

kw1, p3 = kruskal(*race_g)
print("Kruskal-Wallis edad vs raza: H =", round(kw1,3), "  p =", p3)
###

###
outcome_df = ds[['stop_outcome', 'driver_age']].dropna()
outcome_g = [a['driver_age'].values for _, a in outcome_df.groupby('stop_outcome')]

anova2, p4 = f_oneway(*outcome_g)
print("ANOVA edad vs stop outcome: F =", round(anova2,3), "  p =", p4)

kw2, p5 = kruskal(*outcome_g)
print("Kruskal-Wallis edad vs stop outcome: H =", round(kw2,3), "  p =", p5)
###
