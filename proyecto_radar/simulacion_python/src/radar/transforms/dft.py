"""
Módulo: dft.py
--------------
Responsabilidad: Calcular la Transformada Discreta de Fourier utilizando su definición
matemática estricta (fuerza bruta). Toma muestras en el dominio del tiempo y devuelve 
un arreglo de números complejos que representa el dominio de la frecuencia.
Complejidad teórica: O(N^2).
"""

import numpy as np

def compute_dft(samples: np.ndarray) -> np.ndarray:
    """
    Calcula la Transformada Discreta de Fourier (DFT) de un arreglo de entrada 1D.
    Retorna un arreglo de tipo np.complex128.
    """
    
    # Convertimos la entrada a un arreglo de numpy por seguridad.
    x = np.asarray(samples, dtype=float)
    
    # N es el número total de muestras en nuestra señal.
    N = len(x)
    
    # Creamos un arreglo vacío de ceros complejos para almacenar el resultado X[k].
    # np.complex128 garantiza precisión de 64-bits para la parte real y 64 para la imaginaria.
    X = np.zeros(N, dtype=np.complex128)
    
    # Bucle exterior: iteramos sobre cada frecuencia 'k' desde 0 hasta N-1.
    for k in range(N):
        
        # Bucle interior: iteramos sobre cada muestra temporal 'n' desde 0 hasta N-1.
        for n in range(N):
            
            # Ángulo de la exponencial compleja: -j * 2 * pi * k * n / N
            # Usamos 1j nativo de Python para representar el número imaginario 'j'.
            angle = -1j * 2 * np.pi * k * n / N
            
            # Evaluamos e^(ángulo). Por la identidad de Euler, esto es cos(angulo) + j*sin(angulo).
            complex_exp = np.exp(angle)
            
            # Multiplicamos la muestra actual x[n] por la exponencial y la sumamos al acumulador X[k].
            X[k] += x[n] * complex_exp
            
    # Devolvemos el arreglo completo resultante con el espectro complejo.
    return X