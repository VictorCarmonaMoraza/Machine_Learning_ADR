import numpy as np

alt = np.array([1.7, 1.8])

peso = np.array([65, 70])

list_peso = [65,70]

peso = np.array(list_peso)

#Calculamos indice de masa corporal
bmi = peso / alt**2
print(bmi)

#Utilizar slice
imc = np.array([22, 21, 24, 25])

print(imc[2])

print(imc>21)

imc_mayor_21 = imc[imc>21]
print(imc_mayor_21)

#Array 2D
np_2d = np.array([[3,2,4],[2,5,6],[3,1,5]])
print(np_2d)

#Obtener valor central
print(f'El valor central es: {np_2d[1,1]}')  # Fila 1, Columna 1

#Obtner el cuadro 
print(np_2d[1:, 1:])  

print(f'----------------------------------------')
print(np_2d[:,0]) #Obtenr la primera columna

#Obtener la media de la primera columna
print(f'La media de la primera columna es: {np.mean(np_2d[:,0])}')

#Obtener la media de la segunda columna
print(f'La media de la segunda columna es: {np.mean(np_2d[:,1])}')

#Obtenr la mediana de la primera fila
print(np_2d[0,:])
print(f'La mediana de la primera fila es: {np.median(np_2d[0,:])}')
