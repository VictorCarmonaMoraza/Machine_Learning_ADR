
## Ejemplo Iris(150 registros)

#Variables de entrada(atributos)
   ##Longiutd petalo
   ##Ancho petalo
   ##Longitud sepalo
   ##Ancho sepalo

#Variable de salida: Especie
 ##Versicolor
 ##Virginica
 ##Setosa


##Instalamos la libreria de pandas y la importamos(pip install pandas)
import pandas as pd
#Importar algoritmos de la libreria sklearn(pip install scikit-learn)
from sklearn.neighbors import KNeighborsClassifier

#Instalar la libreria de openpyxl para leer archivos de excel(pip install openpyxl)
import openpyxl

#Importar el dataset de iris
df_data = pd.read_excel(r'C:\Users\Victo\Desktop\MachineLearningICA\Unidad3\iris.xlsx')

print(f'------Columnas del dataset ------ ')
#Visualizar el dataset
print(df_data.head())

print(f'------Informacion del dataset ------ ')
#Informacion basica del dataset
print(df_data.info())

##Crear un array (dataframe) con las variables de entrada(X) y otro para la variable 
#de salida(y)
print(f'----Operaciones ------ ')
x = df_data.drop('clase',axis=1).values  # Eliminar la columna 'clase' para obtener las variables de entrada
y = df_data['clase'].values  # Obtener la columna 'clase' como variable de salida

#Dividir datos en conjunto de Training" (ej:80%) y conjunto de "Test"(ej:20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Crear clasificador (Ejemplo KNN)
knn = KNeighborsClassifier(n_neighbors=8)  # Sino ponemos n_neighbors, por defecto es 5

#Entrenamos el modelo
resultado_entreamiento = knn.fit(x_train, y_train)
#Imprimos el entrenamiento
print(f'{resultado_entreamiento}')

#verificamos precision del modelo
print(f'------Precisión del modelo ------ ')
precision = knn.score(x_test, y_test) #Presicion solo en un subconjunto del 20% sin hacer cross validation
print(f'Precisión del modelo: {precision}')

#Predecir resultados de salida(y_prediction) a partir de nuevos datos de entrada(x_new)
df_new  = pd.read_excel(r'C:\Users\Victo\Desktop\MachineLearningICA\Unidad3\iris_nuevos_datos.xlsx')
x_new  = df_new.values  # Convertir el DataFrame a un array de valores

y_prediction = knn.predict(x_new)
print(f'------Predicciones ------ ')
print(f'Predicciones: {y_prediction}'.format(y_prediction))

#Optimizar el modelo
import numpy as np
from sklearn.model_selection import GridSearchCV
# Definir los parámetros a ajustar
param_grid = {'n_neighbors': np.arange(1, 50) } # Probar con valores de 1 a 50 vecinos
knn= KNeighborsClassifier()
# Crear el objeto GridSearchCV
knn_cv = GridSearchCV(knn, param_grid, cv=5)  # cv=5 significa 5-fold cross-validation
# Entrenar el modelo con GridSearchCV
knn_cv.fit(x, y)
# Imprimir los mejores parámetros encontrados
print(f'------Mejores parámetros ------ ')
print(f'Mejores parámetros: {knn_cv.best_params_}')
# Imprimir la mejor precisión obtenida
print(f'Mejores precisión: {knn_cv.best_score_}')

    
print(f'------FLUJO COMPLETO CLASIFICACION------ ')
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Cargar el dataset
df_data =pd.read_excel(r'C:\Users\Victo\Desktop\MachineLearningICA\Unidad3\iris.xlsx')
print(f'{df_data.head()}')

x = df_data.drop('clase', axis=1).values  # Eliminar la columna 'clase' para obtener las variables de entrada
y = df_data['clase'].values
print(f'este dataframe tiene{y}')

#Dividir datos en conjunto de Training" (ej:80%) y conjunto de "Test"(ej:20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Construir modelos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
# Evaluar cada modelo
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42,shuffle=True)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Seleccionar mejor modelo tras brenchmarking
svc = SVC()

#Optimizar modelo
import numpy as np
from sklearn.model_selection import GridSearchCV

Cs =[0.001, 0.01, 0.1,1,10]
gammas = [0.001, 0.01, 0.1,1]
param_grid = {'C': Cs, 'gamma': gammas}
svc_cv = GridSearchCV(svc, param_grid, cv=5)  # cv=5 significa 5-fold cross-validation
svc_cv.fit(x, y)
print(f' Datos: {svc_cv.best_params_}')
print(svc_cv.best_score_)

#Predecir esultados de salida(y_prediction) a partir de nuevos datos de entrada(x_new)
df_new = pd.read_excel(r'C:\Users\Victo\Desktop\MachineLearningICA\Unidad3\iris_nuevos_datos.xlsx')
x_new = df_new.values  # Convertir el DataFrame a un array de valores