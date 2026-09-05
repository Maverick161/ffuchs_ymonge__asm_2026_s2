"""
Módulo: correlation.py
----------------------
Responsabilidad: Calcular la similitud entre la señal transmitida y la recibida
para diferentes desplazamientos temporales (lags).
"""

import numpy as np
from dataclasses import dataclass
from radar.signals.signal import Signal

@dataclass
class CorrelationResult:
    """Contenedor estructurado para los resultados de una correlación."""
    values: np.ndarray
    lags: np.ndarray
    sample_rate: float
    method: str

def cross_correlate_direct(tx_signal: Signal, rx_signal: Signal) -> CorrelationResult:
    """
    Implementa la correlación cruzada discreta de forma directa O(N^2).
    Calcula: r_yx[k] = sum(y[n] * x[n-k])
    
    Convención de Lag: Un lag positivo indica que la señal recibida (RX) 
    está atrasada respecto a la señal transmitida (TX).
    """
    if tx_signal.sample_rate != rx_signal.sample_rate:
        raise ValueError("Las señales deben tener la misma frecuencia de muestreo.")
        
    # np.correlate con mode='full' realiza el cálculo O(N^2) completo.
    # Al pasar (rx, tx), garantizamos que un retraso en rx produzca un pico en un lag positivo.
    corr_values = np.correlate(rx_signal.samples, tx_signal.samples, mode='full')
    
    # Construimos el eje de lags (desplazamientos)
    nx = tx_signal.number_of_samples
    ny = rx_signal.number_of_samples
    
    # En mode='full', el lag va desde -(Nx - 1) hasta (Ny - 1)
    lags = np.arange(-(nx - 1), ny)
    
    return CorrelationResult(
        values=corr_values,
        lags=lags,
        sample_rate=tx_signal.sample_rate,
        method="direct"
    )