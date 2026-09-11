"""
Módulo: signal.py
-----------------
Responsabilidad: Definir la estructura de datos base para cualquier señal en el sistema.
Al encapsular las muestras, la frecuencia de muestreo y calcular automáticamente
el vector de tiempo, evitamos desajustes entre la señal y su eje temporal en 
experimentos posteriores.
"""

import numpy as np  # Importamos numpy para el manejo eficiente de arreglos numéricos
from dataclasses import dataclass  # Importamos dataclass para crear clases contenedoras de datos sin escribir mucho código

@dataclass  # Este decorador genera automáticamente métodos especiales en la clase como __init__ (constructor)
class Signal:
    """Representación integral de una señal discreta en el tiempo."""
    
    samples: np.ndarray  # Arreglo unidimensional que contiene las amplitudes numéricas de la señal
    sample_rate: float  # Frecuencia de muestreo (Fs) en Hertz (cuántas muestras se toman por segundo)
    name: str = "Signal"  # Nombre identificador para facilitar las leyendas en las gráficas (por defecto "Signal")

    @property  # Este decorador permite acceder a un método como si fuera una variable/atributo de la clase
    def number_of_samples(self) -> int:
        """Devuelve el tamaño 'N' de la señal."""
        return len(self.samples)  # Retorna la longitud del arreglo de muestras numpy

    @property  # Decorador para calcular propiedades al vuelo
    def duration(self) -> float:
        """Devuelve la duración total de la señal en segundos."""
        return self.number_of_samples / self.sample_rate  # Divide muestras totales entre muestras por segundo (N/Fs)

    @property  # Decorador para generar el eje temporal solo cuando se solicite
    def time_vector(self) -> np.ndarray:
        """Devuelve el eje X temporal correspondiente a las muestras."""
        # Genera un arreglo desde 0 hasta el número de muestras, y lo divide por Fs para pasarlo a segundos
        return np.arange(self.number_of_samples) / self.sample_rate