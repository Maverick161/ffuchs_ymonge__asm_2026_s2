"""
Módulo: fft.py
--------------
Responsabilidad: Calcular la Transformada Rápida de Fourier utilizando el algoritmo
Cooley-Tukey Radix-2 de forma recursiva. Mejora drásticamente el rendimiento sobre 
la DFT convencional, reduciendo la complejidad a O(N log N).
Requiere estrictamente que el tamaño de la entrada N sea una potencia de 2.
"""

import numpy as np
from radar.utils.validation import is_power_of_two  # Importamos nuestro validador

def compute_fft(samples: np.ndarray) -> np.ndarray:
    """
    Calcula la FFT Radix-2 de un arreglo 1D de forma recursiva.
    Si la entrada no es potencia de 2, lanza un error para proteger el cálculo.
    """
    
    # Aseguramos que trabajamos con un arreglo de numpy
    x = np.asarray(samples, dtype=float)
    N = len(x)
    
    # Condición de seguridad inicial: el algoritmo Radix-2 exige que N sea potencia de 2
    if not is_power_of_two(N):
        raise ValueError(f"El tamaño de la señal ({N}) no es potencia de 2. Usa pad_to_power_of_two primero.")
        
    # CASO BASE: Si la señal tiene tamaño 1, su transformada de Fourier es ella misma.
    # Aquí termina la recursividad y empieza a devolverse hacia arriba.
    if N <= 1:
        return x.astype(np.complex128)
        
    # PASO 1 (Divide): Separamos las muestras en índices pares e impares.
    # x[0::2] toma desde el índice 0 saltando de 2 en 2 (0, 2, 4...)
    # x[1::2] toma desde el índice 1 saltando de 2 en 2 (1, 3, 5...)
    even_samples = x[0::2]
    odd_samples = x[1::2]
    
    # PASO 2 (Vencerás): Llamamos recursivamente a la misma función para cada mitad.
    X_even = compute_fft(even_samples)
    X_odd = compute_fft(odd_samples)
    
    # PASO 3 (Combinar): Calculamos los Twiddle Factors (Factores de giro) para esta etapa.
    # Solo necesitamos calcular la mitad (N/2) gracias a la simetría del plano complejo.
    k = np.arange(N // 2)
    # Factor de giro: e^(-j * 2pi * k / N)
    twiddle_factors = np.exp(-1j * 2 * np.pi * k / N)
    
    # Preparamos el arreglo vacío para el resultado de este nivel
    X = np.zeros(N, dtype=np.complex128)
    
    # Reconstruimos la señal (Efecto Mariposa o Butterfly operation):
    # La primera mitad del espectro usa suma (+).
    X[0 : N//2] = X_even + twiddle_factors * X_odd
    
    # La segunda mitad del espectro usa resta (-), explotando la periodicidad.
    X[N//2 : N] = X_even - twiddle_factors * X_odd
    
    # Retornamos el espectro combinado a la etapa anterior
    return X