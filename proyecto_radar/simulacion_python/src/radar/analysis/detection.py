"""
Módulo: detection.py
--------------------
Responsabilidad: Interpretar los resultados de correlación para estimar
el retardo de los ecos (en muestras y en segundos).
"""

import numpy as np
import math
from radar.analysis.correlation import CorrelationResult

def find_peak(corr_result: CorrelationResult) -> int:
    """
    Encuentra el lag correspondiente al pico de máxima correlación (valor absoluto).
    """
    if len(corr_result.values) == 0:
        raise ValueError("El resultado de correlación está vacío.")
        
    if len(corr_result.values) != len(corr_result.lags):
        raise ValueError("Inconsistencia: Los valores y los lags tienen longitudes diferentes.")
        
    # Verificar si el arreglo es completamente nulo (todos ceros)
    if not np.any(corr_result.values):
        raise ValueError("El resultado de correlación es completamente nulo. No hay picos.")
        
    # Buscamos el índice donde ocurre el máximo valor absoluto de correlación
    peak_idx = int(np.argmax(np.abs(corr_result.values)))
    
    # Devolvemos el retardo (lag) físico asociado a ese índice
    return int(corr_result.lags[peak_idx])

def estimate_delay(corr_result: CorrelationResult) -> tuple[int, float]:
    """
    Estima el retardo del eco principal detectado.
    
    Retorna:
        (delay_samples, delay_seconds)
    """
    fs = corr_result.sample_rate
    
    # Blindaje matemático: rechazar <= 0, NaN e Infinitos
    if math.isnan(fs) or math.isinf(fs) or fs <= 0.0:
        raise ValueError(f"Frecuencia de muestreo inválida para estimar tiempo: {fs}")
        
    delay_samples = find_peak(corr_result)
    delay_seconds = float(delay_samples) / fs
    
    return delay_samples, delay_seconds