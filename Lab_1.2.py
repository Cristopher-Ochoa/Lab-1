# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:52:27 2024

@author: Santiago
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import wfdb

# Ruta a los archivos .dat y .hea (sin extensión)
record_name = 'rec_1'  

# Leer el registro
try:
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, 0]  # Asumiendo que la señal de interés está en el primer canal
except Exception as e:
    print(f"Error leyendo el registro: {e}")

# Calcular la media, la desviación estándar y el coeficiente de variación utilizando diferentes métodos
try:
    # Método 1: Usando numpy
    mean_np = np.mean(signal)
    std_np = np.std(signal)
    cv_np = (std_np / mean_np) * 100 if mean_np != 0 else float('inf')

    # Método 2: Usando pandas
    signal_series = pd.Series(signal)
    mean_pd = signal_series.mean()
    std_pd = signal_series.std()
    cv_pd = (std_pd / mean_pd) * 100 if mean_pd != 0 else float('inf')

    # Método 3: Usando scipy
    from scipy.stats import variation
    mean_sp = np.mean(signal)  # scipy no tiene una función directa para la media
    std_sp = np.std(signal, ddof=1)  # ddof=1 para usar el estimador de muestra
    cv_sp = variation(signal) * 100

    print("Usando numpy:")
    print(f"Media: {mean_np}")
    print(f"Desviación estándar: {std_np}")
    print(f"Coeficiente de variación: {cv_np}%")

    print("\nUsando pandas:")
    print(f"Media: {mean_pd}")
    print(f"Desviación estándar: {std_pd}")
    print(f"Coeficiente de variación: {cv_pd}%")

    print("\nUsando scipy:")
    print(f"Media: {mean_sp}")
    print(f"Desviación estándar: {std_sp}")
    print(f"Coeficiente de variación: {cv_sp}%")
except Exception as e:
    print(f"Error calculando la media, la desviación estándar o el coeficiente de variación: {e}")

# Graficar la señal
try:
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Señal')
    plt.axhline(y=mean_np, color='r', linestyle='--', label=f'Media: {mean_np:.2f}')
    plt.axhline(y=mean_np + std_np, color='g', linestyle='--', label=f'Media + 1σ: {mean_np + std_np:.2f}')
    plt.axhline(y=mean_np - std_np, color='g', linestyle='--', label=f'Media - 1σ: {mean_np - std_np:.2f}')
    plt.title('Señal')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error graficando la señal: {e}")

# Graficar el histograma y la función de probabilidad para la señal
try:
    plt.figure(figsize=(10, 4))
    count, bins, ignored = plt.hist(signal, bins=30, density=True, alpha=0.6, color='b', label='Histograma')
    pdf = norm.pdf(bins, mean_np, std_np)
    plt.plot(bins, pdf, 'k', linewidth=2, label='Función de probabilidad')
    plt.title('Histograma y función de probabilidad')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error graficando histograma y función de probabilidad: {e}")