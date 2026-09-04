"""
Módulo: fft.py
Responsabilidad: Implementación recursiva de la FFT Radix-2 (Cooley-Tukey).
Dependencias: NumPy, is_power_of_two (desde utils.validation).
"""

import numpy as np
from radar.utils.validation import is_power_of_two


def compute_fft(samples: np.ndarray) -> np.ndarray:
    """
    Calcula la FFT Radix-2 de forma recursiva para arreglos 1D.

    Args:
        samples: Arreglo de entrada (real o complejo).

    Returns:
        Espectro en frecuencia (complejo).

    Raises:
        ValueError: Si el tamaño no es potencia de 2.
    """
    # Convierte a array y valida estructura
    x_raw = np.asarray(samples)

    if x_raw.size == 0:
        raise ValueError("La señal no puede estar vacía.")
    if x_raw.ndim != 1:
        raise ValueError(f"Se requiere arreglo 1D, recibido {x_raw.ndim}D.")

    # Cast a complejo para preservar la parte imaginaria
    x = x_raw.astype(np.complex128)
    N = len(x)

    # La FFT Radix-2 exige longitud potencia de 2
    if not is_power_of_two(N):
        raise ValueError(f"El tamaño {N} no es potencia de 2.")

    # Caso base: señal de 1 muestra (su DFT es ella misma)
    if N <= 1:
        return x

    # Divide en pares e impares (decimación en tiempo)
    even_samples = x[0::2]
    odd_samples = x[1::2]

    # Calcula FFT recursivamente para cada mitad
    X_even = compute_fft(even_samples)
    X_odd = compute_fft(odd_samples)

    # Factores twiddle: W_N^k = exp(-j*2π*k/N)
    k = np.arange(N // 2)
    twiddle_factors = np.exp(-1j * 2 * np.pi * k / N)

    # Combina ambas mitades (mariposa)
    X = np.zeros(N, dtype=np.complex128)
    X[0 : N//2] = X_even + twiddle_factors * X_odd
    X[N//2 : N] = X_even - twiddle_factors * X_odd

    return X