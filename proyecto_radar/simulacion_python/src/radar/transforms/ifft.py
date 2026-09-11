"""
Módulo: ifft.py
---------------
Responsabilidad: Calcular la Transformada Inversa Rápida de Fourier (IFFT)
reutilizando la implementación existente de la FFT Radix-2.
"""

import numpy as np
from radar.transforms.fft import compute_fft

def compute_ifft(complex_spectrum: np.ndarray) -> np.ndarray:
    """
    Calcula la IFFT utilizando la relación matemática:
    x[n] = (1/N) * conj( FFT( conj(X[k]) ) )
    """
    X = np.asarray(complex_spectrum, dtype=np.complex128)
    N = len(X)
    
    if N == 0:
        raise ValueError("El espectro no puede estar vacío.")
        
    # 1. Conjugado de la entrada
    X_conj = np.conj(X)
    
    # 2. FFT del conjugado (reutilizando nuestro módulo validado)
    x_fft = compute_fft(X_conj)
    
    # 3. Conjugado del resultado y normalización por N
    x = np.conj(x_fft) / N
    
    return x