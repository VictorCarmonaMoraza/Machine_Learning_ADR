#El elemento principla de pandas es el dataframe

import pandas as pd


df = pd.read_excel(r'C:\Users\Victo\Desktop\MachineLearningICA\recursos\Ventas.xlsx')

print(df)

#Solo queremos quedarnos con las columnas de Pais,precio y Territorio
df_nuevo = df[['País', 'Precio Unitario', 'Territorio']]
print(df_nuevo)
#Obtenemos numero de filas y columnas
print(f'Tiene elementos duplicados {df_nuevo.shape}')

#Obtener los registros del uno al tres
print(df_nuevo[1:3])

#Elimianr  duplicados
df_nuevo = df_nuevo.drop_duplicates()
#Obtenemos el numero de filas y columnas
print(f'Sin duplicados {df_nuevo.shape}')

df_nuevo_delete_pais = df_nuevo.drop(columns=['País'])
print(f'Qitando los duplicados de la columna Pais {df_nuevo.shape}')

#Obtener los regitros que contienen Nan
df_nuevo_nan = df_nuevo[df_nuevo.isnull().any(axis=1)]
print(f'Con NaN {df_nuevo_nan.shape}')

#Eliminar los NaN
df_nuevo_sin_Nan = df_nuevo.dropna(inplace=True)
#Eliminar una columna en este caso Precio UNitario
df_nuevo.drop(columns=['Precio Unitario'], axis=1, inplace=True)
print(df_nuevo)

print(f'-----------Filtrar Dataframe------------')
#Filtrar DataFrame
df_filtrado = df_nuevo[df_nuevo['País'] == 'France']
print(f'Filtrado por pais {df_filtrado}')
print(f'Filtrado por pais {df_filtrado.shape}')