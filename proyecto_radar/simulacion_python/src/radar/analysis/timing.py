"""
Módulo: timing.py
-----------------
Responsabilidad: Medir, perfilar y extraer estadísticas de tiempo de ejecución 
para los algoritmos matemáticos, manteniéndolos aislados del experimento.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkResult:
    """Estructura de datos para almacenar el resultado estadístico de un benchmark."""
    algorithm: str
    N: int
    average_time: float
    minimum_time: float
    maximum_time: float
    standard_deviation: float

def benchmark_algorithm(algorithm_name: str, func: Callable, data: np.ndarray, repetitions: int = 10) -> BenchmarkResult:
    """
    Ejecuta repetidamente un algoritmo y extrae sus estadísticas de rendimiento.
    """
    times = []
    
    # Ejecución del perfilado
    for _ in range(repetitions):
        start = time.perf_counter()
        func(data)  # Llamada al algoritmo (ej. compute_dft o compute_fft)
        end = time.perf_counter()
        times.append(end - start)
        
    times = np.array(times)
    
    return BenchmarkResult(
        algorithm=algorithm_name,
        N=len(data),
        average_time=float(np.mean(times)),
        minimum_time=float(np.min(times)),
        maximum_time=float(np.max(times)),
        standard_deviation=float(np.std(times))
    )