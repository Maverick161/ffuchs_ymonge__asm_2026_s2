"""
Módulo: generators.py
Responsabilidad: Generar señales acústicas sintéticas (senoides y ruido) para pruebas.
Dependencias: NumPy, Signal (DTO del dominio).
"""

import numpy as np
from typing import Optional
from radar.signals.signal import Signal


def generate_sine(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> Signal:
    """
    Genera una señal senoidal pura.

    Args:
        frequency: Frecuencia en Hz.
        sample_rate: Muestras por segundo (>0).
        duration: Segundos (>0).
        amplitude: Amplitud pico.
        phase: Fase inicial en radianes.

    Returns:
        Signal con las muestras generadas.
    """
    # Validación de parámetros físicos (evita división por cero o arrays vacíos)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate debe ser > 0. Recibido: {sample_rate}")
    if duration <= 0:
        raise ValueError(f"duration debe ser > 0. Recibido: {duration}")

    # Construcción del eje temporal discreto
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Síntesis de la onda
    samples = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return Signal(samples=samples, sample_rate=sample_rate, name=f"Sine_{frequency}Hz")


def add_noise(
    signal: Signal,
    noise_variance: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> Signal:
    """
    Añade ruido blanco gaussiano (AWGN) a una señal existente.

    Args:
        signal: Señal limpia.
        noise_variance: Varianza del ruido (>=0).
        rng: Generador inyectado para garantizar reproducibilidad en tests.

    Returns:
        Signal con el ruido superpuesto.
    """
    # La varianza negativa no tiene sentido físico ni matemático
    if noise_variance < 0:
        raise ValueError(f"noise_variance no puede ser negativa. Recibido: {noise_variance}")

    std_dev = np.sqrt(noise_variance)

    # Generación del ruido (usa el generador inyectado si existe, para tests deterministas)
    if rng is None:
        noise = np.random.normal(0, std_dev, signal.number_of_samples)
    else:
        noise = rng.normal(0, std_dev, signal.number_of_samples)

    # Superposición aditiva (señal + ruido)
    noisy_samples = signal.samples + noise

    return Signal(
        samples=noisy_samples,
        sample_rate=signal.sample_rate,
        name=f"{signal.name}_Noisy"
    )