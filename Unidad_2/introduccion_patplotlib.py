import matplotlib.pyplot as plt

# Datos de ejemplo
ejex = [1990, 1995, 2000, 2030]
ejey = [5, 5.5, 6, 8]

print(f'----------------Line Plot----------------')
plt.plot(ejex, ejey)
plt.show()


print(f'----------------Scatter Plot----------------')
plt.scatter(ejex, ejey)
plt.show()


print(f'----------------Histogram Plot----------------')
temperatura = [12, 12.3, 13.5, 14, 13, 14, 5, 17]
plt.hist(temperatura, bins=5, edgecolor='black')
plt.show()

print(f'----------------CDF Cumulative Distribution Function----------------')
plt.hist(temperatura, bins=10, edgecolor='black',cumulative=True)
plt.show()
