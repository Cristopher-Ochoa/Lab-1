# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:05:27 2024

@author: Santiago
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import wfdb

# Ruta a los archivos .dat y .hea (sin extensión)
record_name = 'rec_1'  # Reemplaza con el nombre base de tu archivo (sin extensión)

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

# Función para calcular SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Contaminar la señal con ruido gaussiano y calcular SNR
try:
    noise_std = 0.5 * std_np  # Ajusta el nivel de ruido aquí
    noise_gaussian = np.random.normal(0, noise_std, signal.shape)
    noisy_signal_gaussian = signal + noise_gaussian

    snr_gaussian = calculate_snr(signal, noise_gaussian)
    print(f"SNR después de añadir ruido gaussiano: {snr_gaussian:.2f} dB")

    # Graficar la señal original y la señal con ruido gaussiano
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Señal original')
    plt.plot(noisy_signal_gaussian, label='Señal con ruido gaussiano')
    plt.title('Señal original y señal con ruido gaussiano')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar el histograma y la función de probabilidad para la señal con ruido gaussiano
    plt.figure(figsize=(10, 4))
    count, bins, ignored = plt.hist(noisy_signal_gaussian, bins=30, density=True, alpha=0.6, color='b', label='Histograma')
    pdf = norm.pdf(bins, mean_np, std_np + noise_std)
    plt.plot(bins, pdf, 'k', linewidth=2, label='Función de probabilidad')
    plt.title('Histograma y función de probabilidad (señal con ruido gaussiano)')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error al añadir ruido gaussiano o calcular SNR: {e}")

# Contaminar la señal con ruido de impulso y calcular SNR
try:
    impulse_noise = np.zeros_like(signal)
    num_impulses = int(0.01 * len(signal))  # Ajusta el porcentaje de impulsos aquí
    impulse_positions = np.random.choice(len(signal), num_impulses, replace=False)
    impulse_magnitude = 5 * std_np  # Ajusta la magnitud del impulso aquí
    impulse_noise[impulse_positions] = impulse_magnitude
    noisy_signal_impulse = signal + impulse_noise

    snr_impulse = calculate_snr(signal, impulse_noise)
    print(f"SNR después de añadir ruido de impulso: {snr_impulse:.2f} dB")

    # Graficar la señal original y la señal con ruido de impulso
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Señal original')
    plt.plot(noisy_signal_impulse, label='Señal con ruido de impulso')
    plt.title('Señal original y señal con ruido de impulso')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar el histograma y la función de probabilidad para la señal con ruido de impulso
    plt.figure(figsize=(10, 4))
    count, bins, ignored = plt.hist(noisy_signal_impulse, bins=30, density=True, alpha=0.6, color='b', label='Histograma')
    pdf = norm.pdf(bins, mean_np, std_np)
    plt.plot(bins, pdf, 'k', linewidth=2, label='Función de probabilidad')
    plt.title('Histograma y función de probabilidad (señal con ruido de impulso)')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error al añadir ruido de impulso o calcular SNR: {e}")

# Contaminar la señal con ruido tipo artefacto y calcular SNR
try:
    artifact_noise = np.zeros_like(signal)
    num_artifacts = int(0.01 * len(signal))  # Ajusta el porcentaje de artefactos aquí
    artifact_positions = np.random.choice(len(signal), num_artifacts, replace=False)
    artifact_magnitude = 10 * std_np  # Ajusta la magnitud del artefacto aquí
    artifact_noise[artifact_positions] = artifact_magnitude
    noisy_signal_artifact = signal + artifact_noise

    snr_artifact = calculate_snr(signal, artifact_noise)
    print(f"SNR después de añadir ruido tipo artefacto: {snr_artifact:.2f} dB")

    # Graficar la señal original y la señal con ruido tipo artefacto
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Señal original')
    plt.plot(noisy_signal_artifact, label='Señal con ruido tipo artefacto')
    plt.title('Señal original y señal con ruido tipo artefacto')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar el histograma y la función de probabilidad para la señal con ruido tipo artefacto
    plt.figure(figsize=(10, 4))
    count, bins, ignored = plt.hist(noisy_signal_artifact, bins=30, density=True, alpha=0.6, color='b', label='Histograma')
    pdf = norm.pdf(bins, mean_np, std_np)
    plt.plot(bins, pdf, 'k', linewidth=2, label='Función de probabilidad')
    plt.title('Histograma y función de probabilidad (señal con ruido tipo artefacto)')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error al añadir ruido tipo artefacto o calcular SNR: {e}")

# Graficar media, desviación estándar y coeficiente de variación
try:
    metrics = ['numpy', 'pandas', 'scipy']
    means = [mean_np, mean_pd, mean_sp]
    stds = [std_np, std_pd, std_sp]
    cvs = [cv_np, cv_pd, cv_sp]

    # Graficar media
    plt.figure(figsize=(10, 4))
    plt.bar(metrics, means, color=['blue', 'orange', 'green'])
    plt.title('Media de la señal por diferentes métodos')
    plt.xlabel('Método')
    plt.ylabel('Media')
    plt.grid(True)
    plt.show()

    # Graficar desviación estándar
    plt.figure(figsize=(10, 4))
    plt.bar(metrics, stds, color=['blue', 'orange', 'green'])
    plt.title('Desviación estándar de la señal por diferentes métodos')
    plt.xlabel('Método')
    plt.ylabel('Desviación estándar')
    plt.grid(True)
    plt.show()

    # Graficar coeficiente de variación
    plt.figure(figsize=(10, 4))
    plt.bar(metrics, cvs, color=['blue', 'orange', 'green'])
    plt.title('Coeficiente de variación de la señal por diferentes métodos')
    plt.xlabel('Método')
    plt.ylabel('Coeficiente de variación (%)')
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error graficando las métricas: {e}")