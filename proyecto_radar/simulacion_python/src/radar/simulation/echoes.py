"""
Módulo: echoes.py
-----------------
Responsabilidad: Simular el canal acústico recibiendo una señal transmitida y 
generando ecos con retardos (delay_samples) y atenuación, modelando un sistema LTI.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from radar.signals.signal import Signal
from radar.signals.generators import add_noise

@dataclass
class EchoConfig:
    """Configuración física de un eco individual."""
    delay_samples: int
    attenuation: float

def simulate_echoes(transmitted_signal: Signal, echoes: List[EchoConfig], noise_variance: float = 0.0, rng: Optional[np.random.Generator] = None) -> Signal:
    """
    Genera la señal recibida (RX) superponiendo múltiples ecos de la señal original.
    La longitud de la señal RX se ajusta automáticamente para contener el eco más lejano.
    """
    if not echoes:
        raise ValueError("Debe proveer al menos una configuración de eco (EchoConfig).")
        
    if noise_variance < 0.0:
        raise ValueError(f"La varianza del ruido no puede ser negativa. Se recibió: {noise_variance}")
        
    max_delay = 0
    for i, echo in enumerate(echoes):
        # Blindaje matemático: rechazar retardos fraccionarios explícitamente
        if int(echo.delay_samples) != echo.delay_samples:
            raise ValueError(f"El eco {i} tiene un retardo fraccionario no soportado: {echo.delay_samples}")
            
        if echo.delay_samples < 0:
            raise ValueError(f"El eco {i} tiene un retardo negativo: {echo.delay_samples}")
            
        if echo.attenuation < 0.0:
            raise ValueError(f"El eco {i} tiene atenuación negativa: {echo.attenuation}")
            
        max_delay = max(max_delay, int(echo.delay_samples))
        
    tx_len = transmitted_signal.number_of_samples
    rx_len = tx_len + max_delay
    rx_samples = np.zeros(rx_len)
    
    # Superposición lineal de ecos
    for echo in echoes:
        start_idx = int(echo.delay_samples)
        end_idx = start_idx + tx_len
        rx_samples[start_idx:end_idx] += transmitted_signal.samples * echo.attenuation
        
    rx_signal = Signal(
        samples=rx_samples, 
        sample_rate=transmitted_signal.sample_rate, 
        name=f"{transmitted_signal.name}_RX"
    )
    
    if noise_variance > 0.0:
        rx_signal = add_noise(rx_signal, noise_variance=noise_variance, rng=rng)
        
    return rx_signal