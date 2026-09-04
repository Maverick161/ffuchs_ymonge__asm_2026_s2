"""
Módulo: validation.py
---------------------
Responsabilidad: Proveer funciones auxiliares para validar restricciones matemáticas
sobre los datos numéricos, específicamente relacionadas con el tamaño de las muestras (N)
y las potencias de base 2 requeridas por algoritmos como la FFT Radix-2.
"""

import numpy as np
from radar.signals.signal import Signal

def is_power_of_two(n: int) -> bool:
    """Verifica si un número entero positivo es potencia de 2."""
    
    # Verificamos que 'n' sea mayor a cero.
    if n <= 0:
        return False
    # Una propiedad de los números binarios: si 'n' es potencia de 2, la operación 
    # AND bit a bit (n & (n - 1)) siempre da 0. Ej: 8 (1000) & 7 (0111) = 0000.
    return (n & (n - 1)) == 0

def next_power_of_two(n: int) -> int:
    """Calcula la potencia de 2 inmediatamente superior o igual a 'n'."""
    
    # Si 'n' ya es potencia de 2, lo devolvemos sin cambios.
    if is_power_of_two(n):
        return n
    
    # np.log2 calcula el logaritmo base 2. np.ceil redondea hacia arriba al entero más cercano.
    # Por ejemplo, para n=1000: log2(1000)=9.96 -> ceil=10 -> 2**10 = 1024.
    p = np.ceil(np.log2(n))
    
    # Retornamos 2 elevado a esa potencia entera calculada.
    return int(2 ** p)

def pad_to_power_of_two(signal: Signal) -> Signal:
    """Rellena la señal con ceros (Zero-Padding) hasta la siguiente potencia de 2."""
    
    # Obtenemos el tamaño original de la señal (N).
    n = signal.number_of_samples
    
    # Verificamos si ya cumple la condición; si es así, devolvemos la señal tal cual.
    if is_power_of_two(n):
        return signal
        
    # Calculamos cuál es el tamaño objetivo (N_target) que es potencia de 2.
    target_n = next_power_of_two(n)
    
    # Calculamos cuántos ceros faltan para llegar al tamaño objetivo.
    zeros_to_add = target_n - n
    
    # Creamos un arreglo de ceros utilizando numpy.
    zero_padding = np.zeros(zeros_to_add)
    
    # Concatenamos (unimos) las muestras originales con el arreglo de ceros al final.
    padded_samples = np.concatenate((signal.samples, zero_padding))
    
    # Empaquetamos todo en una nueva Signal, manteniendo el Fs original y actualizando el nombre.
    return Signal(samples=padded_samples, 
                  sample_rate=signal.sample_rate, 
                  name=f"{signal.name}_Padded")