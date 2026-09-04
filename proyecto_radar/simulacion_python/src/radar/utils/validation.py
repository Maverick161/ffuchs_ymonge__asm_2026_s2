"""
Módulo: validation.py
Responsabilidad: Funciones de seguridad matemática (potencias de 2 y zero-padding).
Dependencias: NumPy, Signal.
"""

import numpy as np
from radar.signals.signal import Signal


def is_power_of_two(n: int) -> bool:
    """Verifica si un entero positivo es potencia de 2."""
    if n <= 0:
        return False
    # Truco bit a bit: potencias de 2 tienen un solo bit encendido.
    return (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """
    Calcula la potencia de 2 inmediatamente superior o igual a 'n'.
    
    Args:
        n: Entero positivo.
    
    Returns:
        Potencia de 2 calculada.
    """
    # log2 solo está definido para valores > 0
    if n <= 0:
        raise ValueError(f"n debe ser > 0. Recibido: {n}")

    if is_power_of_two(n):
        return n

    # Redondeo al techo en el dominio logarítmico
    p = np.ceil(np.log2(n))
    return int(2 ** p)


def pad_to_power_of_two(signal: Signal) -> Signal:
    """
    Rellena la señal con ceros (Zero-Padding) hasta la siguiente potencia de 2.
    
    Útil para preparar señales para algoritmos FFT, que son más eficientes
    con longitudes potencias de 2.

    Args:
        signal: Señal de entrada.

    Returns:
        Nueva señal con ceros añadidos al final.
    """
    n = signal.number_of_samples

    # Si ya es potencia de 2, se retorna la señal sin modificar
    if is_power_of_two(n):
        return signal

    target_n = next_power_of_two(n)
    zeros_to_add = target_n - n

    # Añade ceros al final de la muestra
    zero_padding = np.zeros(zeros_to_add)
    padded_samples = np.concatenate((signal.samples, zero_padding))

    return Signal(
        samples=padded_samples,
        sample_rate=signal.sample_rate,
        name=f"{signal.name}_Padded"
    )