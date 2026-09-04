"""
Módulo: generators.py
---------------------
Responsabilidad: Generar señales acústicas sintéticas para realizar pruebas en software.
Las funciones de este módulo construyen matemáticamente las señales (evaluando senos
y distribuciones aleatorias) y las devuelven empaquetadas en nuestro objeto `Signal`.
"""

import numpy as np  # Importamos numpy para operaciones matemáticas vectorizadas (más rápidas que bucles for)
from radar.signals.signal import Signal  # Importamos la estructura de datos que creamos en el archivo anterior

def generate_sine(frequency: float, sample_rate: float, duration: float, amplitude: float = 1.0, phase: float = 0.0) -> Signal:
    """Genera una señal senoidal pura."""
    
    # Calculamos el número exacto de muestras multiplicando duración (s) por frecuencia de muestreo (Hz)
    num_samples = int(sample_rate * duration)
    
    # Creamos el eje temporal discreto: un arreglo de 0 a num_samples-1, escalado a segundos
    t = np.arange(num_samples) / sample_rate
    
    # Evaluamos la ecuación matemática de la onda: x(t) = A * sin(2 * pi * f * t + phi)
    samples = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Empaquetamos y retornamos el resultado utilizando nuestra clase base
    return Signal(samples=samples, sample_rate=sample_rate, name=f"Sine_{frequency}Hz")

def add_noise(signal: Signal, noise_variance: float = 0.1) -> Signal:
    """Añade ruido blanco gaussiano a una señal existente."""
    
    # Calculamos la desviación estándar como la raíz cuadrada de la varianza solicitada
    std_dev = np.sqrt(noise_variance)
    
    # Generamos valores aleatorios con distribución normal (media 0, desviación calculada) del mismo tamaño que la señal
    noise = np.random.normal(0, std_dev, signal.number_of_samples)
    
    # Sumamos el ruido vectorizado a las muestras originales
    noisy_samples = signal.samples + noise
    
    # Retornamos una nueva señal contaminada con el ruido, preservando la frecuencia de muestreo original
    return Signal(samples=noisy_samples, sample_rate=signal.sample_rate, name=f"{signal.name}_Noisy")