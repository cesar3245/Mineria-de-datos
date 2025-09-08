import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

binarias  = [i for i in ds5.columns if ds5[i].dropna().nunique() == 2]
numericas = [i for i in ds5.select_dtypes(include=[np.number]).columns if i not in binarias]

base = ds5[numericas].agg(['mean', 'median', 'var', 'std', 'min', 'max']).T
base['range'] = base['max'] - base['min']
mode_v = ds5[numericas].apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)

tabla_num = (base.drop(columns=['min','max']).assign(Mode=mode_v)[['mean', 'median','Mode', 'var', 'std', 'range']].rename(columns={'mean': 'Mean', 'median': 'Median', 'var': 'Variance', 'std': 'Std Dev', 'range': 'Range'}).round(3))

tabla_bin = pd.DataFrame({'Proportion': ds5[binarias].mean(), 'Count_1s'  : ds5[binarias].sum()})

def cats(name):
    if name.startswith("stop_outcome"):
        return "Stop Outcome"
    elif name.startswith("stop_duration"):
        return "Stop Duration"
    elif name.startswith("violation_raw"):
        return "Violation Raw"
    elif name.startswith("violation_"):
        return "Violation"
    elif name.startswith("driver_race"):
        return "Driver Race"
    elif name.startswith("driver_"):
        return "Driver"
    elif name.startswith("stop_wday"):
        return "Stop Weekday"
    else:
        return "Other"

tabla_bin["Category"] = tabla_bin.index.map(cats)

tabla_bin = tabla_bin.sort_values(["Category", "Count_1s"], ascending=[True, False])

tabla_bin["Proportion"] = (tabla_bin["Proportion"] * 100).round(1)

tabla_bin = tabla_bin[["Category", "Proportion", "Count_1s"]]

#tabla_bin2 = (tabla_bin.style.format({'Proportion': '{:.1f}%', 'Count_1s': '{:,}'}))

tabla_bin["Proportion"] = tabla_bin["Proportion"].map(lambda x: f"{x:.1f}%")
tabla_bin["Count_1s"] = tabla_bin["Count_1s"].map("{:,}".format)

print("Numeric features:")
print(tabla_num)

print("\nBinary fetures:")
print(tabla_bin)
###

###
cols_outcome = [i for i in ds5.columns if i.startswith("stop_outcome_")]

cols_race = ["driver_race_Asian", "driver_race_Black", "driver_race_Hispanic", "driver_race_Other", "driver_race_White"]

cols_actions = ["search_conducted","is_arrested","drugs_related_stop"]

###

###
def resultados_raza(columnas):
    filas = {}
    for i in cols_race:
        a = ds5[i] == 1
        nombre = i.replace("driver_race_", "Race: ")
        if a.any():
            filas[nombre] = ds5.loc[a, columnas].mean()
    return pd.DataFrame(filas).T

resultados2 = resultados_raza(cols_outcome + cols_actions).mul(100).round(1)
resultados3 = resultados2.astype(str) + '%'

print("Outcome by race:")
print(resultados3.to_string())

###

###
resultados_genero = (ds5.groupby("driver_gender")[cols_outcome + cols_actions].mean().mul(100).round(1).rename(index={0:"Female", 1:"Male"}))
resultados_genero2 = resultados_genero.astype(str) + '%'

print("Outcome by gender:")
print(resultados_genero2.to_string())
###

###
det_hora = ds5.groupby("stop_hour").size()
det_hora2 = (det_hora / det_hora.sum() * 100).round(1)
det_hora3 = det_hora2.astype(str) + '%' 

print("Stops by Hour:")
print(det_hora3)
###













