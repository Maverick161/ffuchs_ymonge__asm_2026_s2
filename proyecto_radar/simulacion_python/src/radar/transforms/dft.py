"""
Módulo: dft.py
Responsabilidad: Implementación directa de la Transformada Discreta de Fourier (DFT) según su definición matemática.
Dependencias: NumPy.
"""

import numpy as np


def compute_dft(samples: np.ndarray) -> np.ndarray:
    """
    Calcula la DFT de un arreglo 1D (real o complejo) usando la definición:
    X[k] = Σ_{n=0}^{N-1} x[n] * exp(-j*2π*k*n/N)

    Args:
        samples: Arreglo de entrada.

    Returns:
        Espectro en frecuencia (complejo).
    """
    # Convierte a array y valida estructura
    x_raw = np.asarray(samples)

    if x_raw.size == 0:
        raise ValueError("La señal no puede estar vacía.")
    if x_raw.ndim != 1:
        raise ValueError(f"Se requiere arreglo 1D, recibido {x_raw.ndim}D.")

    # Cast a complejo para preservar la parte imaginaria si la entrada la tuviera
    x = x_raw.astype(np.complex128)
    N = len(x)

    X = np.zeros(N, dtype=np.complex128)

    # Implementación O(N²) por definición (no optimizada, pero exacta)
    for k in range(N):
        for n in range(N):
            angle = -1j * 2 * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)

    return X