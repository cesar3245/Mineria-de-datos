import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, kruskal
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
base = ['driver_age', 'search_conducted', 'drugs_related_stop', 'stop_hour']
dias_semana = [i for i in ds5.columns if i.startswith('stop_wday_')]
raza_condutor = [i for i in ds5.columns if i.startswith('driver_race_')]
X_cols = [i for i in base if i in ds5.columns] + dias_semana + raza_condutor
y = ds5['is_arrested']
X = ds5[X_cols]
###

###
corr = ds5[X_cols + ['is_arrested']].corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de correlacion")
plt.tight_layout(); plt.show()
###

###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
modelo1 = LinearRegression().fit(X_train, y_train)
modelo1_pred = modelo1.predict(X_test)
print("Columnas usadas:", X_cols)
print(f"R^2: {r2_score(y_test, modelo1_pred):.4f}")
###

###
plt.scatter(y_test, modelo1_pred, alpha=0.5)
plt.xlabel("is_arrested"); plt.ylabel("Prediccion")
plt.title("Predicciones")
plt.tight_layout(); plt.show()
###

###
pd.Series(modelo1.coef_, index=X_cols).sort_values(ascending=False).plot(kind='bar')
plt.title("Coeficientes"); plt.ylabel("Peso lineal")
plt.tight_layout(); plt.show()
###
