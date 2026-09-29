"""
Módulo: spectrum.py
-------------------
Responsabilidad: Interpretar los números complejos arrojados por las transformadas (DFT/FFT)
y extraer las componentes físicas: frecuencias, magnitud normalizada y fase.
"""

import numpy as np
from dataclasses import dataclass
from radar.signals.signal import Signal

@dataclass
class Spectrum:
    """Representación estructurada de una señal en el dominio de la frecuencia."""
    frequencies: np.ndarray      # Eje X: Frecuencias en Hertz (bilateral)
    complex_values: np.ndarray   # Los valores arrojados por DFT/FFT
    magnitude: np.ndarray        # Eje Y principal: Amplitud física de cada frecuencia
    phase: np.ndarray            # Eje Y secundario: Ángulo de desfase en radianes
    sample_rate: float           # Frecuencia de muestreo original

    @property
    def positive_frequencies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve la mitad no negativa del espectro (incluyendo DC y Nyquist).
        Para N par, esto abarca exactamente desde el índice 0 hasta N/2.
        Retorna: (frecuencias_positivas, magnitud_positiva, fase_positiva)
        """
        N = len(self.frequencies)
        cutoff = (N // 2) + 1 
        
        # Aplicamos abs() SOLAMENTE aquí para corregir el signo de Nyquist
        # sin alterar el espectro bilateral original del objeto.
        pos_freqs = np.abs(self.frequencies[:cutoff])
        
        return (pos_freqs, self.magnitude[:cutoff], self.phase[:cutoff])

def compute_spectrum(signal: Signal, complex_values: np.ndarray) -> Spectrum:
    """Procesa un espectro complejo para extraer características físicas."""
    
    complex_arr = np.asarray(complex_values)
    
    if complex_arr.ndim != 1:
        raise ValueError(f"El espectro debe ser unidimensional. Se recibió {complex_arr.ndim}D.")
        
    if len(complex_arr) != signal.number_of_samples:
        raise ValueError("Longitud inconsistente entre señal y espectro.")
        
    N = signal.number_of_samples
    fs = signal.sample_rate
    
    # 2. Cálculo del eje de frecuencias bilateral (restaurado a original de NumPy)
    frequencies = np.fft.fftfreq(N, d=1.0/fs)
    
    # 3. Cálculo de la magnitud normalizada
    magnitude = np.abs(complex_arr) / N
    
    if N % 2 == 0:
        # N par: Nyquist está en N/2. Multiplicamos de 1 a N/2-1, y de N/2+1 al final.
        magnitude[1 : N//2] *= 2
        magnitude[(N//2)+1 :] *= 2
    else:
        # N impar: Multiplicamos todo excepto DC (índice 0)
        magnitude[1:] *= 2
        
    # 4. Cálculo de la fase absoluta
    phase = np.angle(complex_arr)
    
    # 5. Limpieza de fase
    threshold = 1e-10
    phase[magnitude < threshold] = 0.0
    
    return Spectrum(
        frequencies=frequencies,
        complex_values=complex_arr,
        magnitude=magnitude,
        phase=phase,
        sample_rate=fs
    )