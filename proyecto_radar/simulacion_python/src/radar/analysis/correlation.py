"""
Módulo: correlation.py
----------------------
Responsabilidad: Calcular la similitud entre la señal transmitida y la recibida
para diferentes desplazamientos temporales (lags).
"""

import numpy as np
from dataclasses import dataclass
from radar.signals.signal import Signal
from radar.utils.validation import next_power_of_two
from radar.transforms.fft import compute_fft
from radar.transforms.ifft import compute_ifft

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

def cross_correlate_fft(tx_signal: Signal, rx_signal: Signal) -> CorrelationResult:
    """
    Implementa la correlación cruzada mediante FFT O(N log N).
    Utiliza zero-padding para evitar aliasing circular y alcanzar
    la longitud lineal necesaria.
    """
    if tx_signal.sample_rate != rx_signal.sample_rate:
        raise ValueError("Las señales deben tener la misma frecuencia de muestreo.")
        
    nx = tx_signal.number_of_samples
    ny = rx_signal.number_of_samples
    
    # 1. Tamaño lineal necesario para convolución/correlación completa
    linear_len = nx + ny - 1
    
    # 2. Siguiente potencia de 2 requerida por nuestra FFT Radix-2
    fft_len = next_power_of_two(linear_len)
    
    # 3. Preparación de señales y Zero Padding
    # Para que la convolución actúe como correlación, TX debe invertirse y conjugarse
    tx_reversed = np.conj(tx_signal.samples[::-1])
    
    rx_padded = np.pad(rx_signal.samples, (0, fft_len - ny))
    tx_padded = np.pad(tx_reversed, (0, fft_len - nx))
    
    # 4. Dominio de la frecuencia
    RX = compute_fft(rx_padded)
    TX = compute_fft(tx_padded)
    
    # 5. Multiplicación espectral (equivale a convolución/correlación en el tiempo)
    R_YX = RX * TX
    
    # 6. Retorno al dominio del tiempo
    r_yx_full = compute_ifft(R_YX)
    
    # 7. Recorte de la longitud útil
    corr_values = r_yx_full[:linear_len]
    
    # Si ambas señales originales eran puramente reales, eliminamos el 
    # residuo imaginario de punto flotante. De lo contrario, lo conservamos.
    if not np.iscomplexobj(tx_signal.samples) and not np.iscomplexobj(rx_signal.samples):
        corr_values = np.real(corr_values)
    
    # 8. Eje de lags (idéntico al de la correlación directa)
    lags = np.arange(-(nx - 1), ny)
    
    return CorrelationResult(
        values=corr_values,
        lags=lags,
        sample_rate=tx_signal.sample_rate,
        method="fft"
    )