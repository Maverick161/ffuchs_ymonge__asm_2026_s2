"""
Módulo: plots.py
----------------
Responsabilidad: Centralizar la generación de gráficas mediante Matplotlib.
Mantiene un formato uniforme (estilo IEEE/ACM) para todas las figuras del 
proyecto, evitando código redundante en los experimentos.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

from radar.signals.signal import Signal
from radar.analysis.spectrum import Spectrum
from radar.analysis.correlation import CorrelationResult
from radar.simulation.echoes import EchoConfig

# Configuración global para estilo académico (para el paper IEEE)
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (8, 4), # Relación de aspecto estándar para papers
    "figure.dpi": 150,        # Alta resolución
    "grid.alpha": 0.5         # Cuadrícula tenue
})

def _save_or_show(fig: plt.Figure, filepath: str = None):
    """Función auxiliar para guardar la figura si se provee una ruta, o mostrarla."""
    if filepath:
        # Asegura que el directorio exista antes de guardar
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_time_domain(signal: Signal, filepath: str = None):
    """
    Grafica una señal en el dominio del tiempo.
    Para señales con muchas muestras, se grafica una línea continua. 
    Para señales muy cortas (ej. < 50), se añaden marcadores para ver las muestras discretas.
    """
    fig, ax = plt.subplots()
    
    # Si la señal es corta, mostramos los puntos discretos
    marker = 'o' if signal.number_of_samples <= 50 else ''
    
    ax.plot(signal.time_vector, signal.samples, marker=marker, color='#1f77b4')
    
    ax.set_title(f"Dominio del Tiempo: {signal.name}")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    
    _save_or_show(fig, filepath)

def plot_magnitude_spectrum(spectrum: Spectrum, filepath: str = None, zoom_freq: float = None):
    """
    Grafica la magnitud del espectro (unilateral).
    """
    fig, ax = plt.subplots()
    
    # Usamos la vista unilateral (positiva) para el análisis físico
    pos_freqs, pos_mags, _ = spectrum.positive_frequencies
    
    # Stem crea líneas verticales ("palitos") ideales para espectros discretos
    markerline, stemlines, baseline = ax.stem(
        pos_freqs, pos_mags, linefmt='b-', markerfmt='bo', basefmt='r-'
    )
    # Ajustamos el grosor de los tallos para estética
    plt.setp(stemlines, 'linewidth', 1.5)
    plt.setp(markerline, 'markersize', 4)
    
    ax.set_title("Espectro de Magnitud")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud Normalizada")
    ax.grid(True)
    
    # Si se solicita, enfocamos la gráfica en una banda específica
    if zoom_freq is not None:
        ax.set_xlim(0, zoom_freq)
        
    # Aseguramos que el eje Y parta de cero para no distorsionar visualmente la amplitud
    ax.set_ylim(bottom=0)
    
    _save_or_show(fig, filepath)

def plot_phase_spectrum(spectrum: Spectrum, filepath: str = None, zoom_freq: float = None):
    """
    Grafica la fase del espectro (unilateral) en grados.
    """
    fig, ax = plt.subplots()
    
    pos_freqs, _, pos_phases = spectrum.positive_frequencies
    
    # Convertimos la fase de radianes a grados para mayor legibilidad en el reporte
    phase_degrees = np.degrees(pos_phases)
    
    markerline, stemlines, baseline = ax.stem(
        pos_freqs, phase_degrees, linefmt='g-', markerfmt='go', basefmt='r-'
    )
    plt.setp(stemlines, 'linewidth', 1.5)
    plt.setp(markerline, 'markersize', 4)
    
    ax.set_title("Espectro de Fase")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Fase (Grados)")
    ax.grid(True)
    
    if zoom_freq is not None:
        ax.set_xlim(0, zoom_freq)
        
    # Limitamos el eje Y al rango estándar de ángulos
    ax.set_ylim(-180, 180)
    # Forzamos ticks espaciados cada 90 grados
    ax.set_yticks([-180, -90, 0, 90, 180])
    
    _save_or_show(fig, filepath)

def plot_execution_times(ns: list[int], dft_times: list[float], fft_times: list[float], filepath: str = None):
    """
    Grafica la comparación de tiempos de ejecución (N vs Tiempo) en escala logarítmica.
    """
    fig, ax = plt.subplots()
    
    ax.plot(ns, dft_times, 'ro-', label=r'DFT $\mathcal{O}(N^2)$')
    ax.plot(ns, fft_times, 'bo-', label=r'FFT $\mathcal{O}(N \log N)$')
    
    ax.set_title("Comparativa de Rendimiento Computacional")
    ax.set_xlabel("Número de Muestras (N)")
    ax.set_ylabel("Tiempo Promedio de Ejecución (s)")
    
    # Escala logarítmica en Y para observar detalles en algoritmos rápidos y lentos
    ax.set_yscale('log')
    # Eje X en base 2 porque nuestras N son potencias de 2
    ax.set_xscale('log', base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    _save_or_show(fig, filepath)

def plot_transmitted_received(tx_signal: Signal, rx_signal: Signal, echoes: List[EchoConfig] = None, filepath: str = None):
    """Grafica la señal transmitida y la señal recibida, marcando los ecos si se proveen."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Señal TX
    ax1.plot(tx_signal.time_vector, tx_signal.samples, color='blue', label="TX (Transmitida)")
    ax1.set_title("Señal Transmitida Original")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    
    # Señal RX
    ax2.plot(rx_signal.time_vector, rx_signal.samples, color='red', alpha=0.8, label="RX (Recibida con Ecos)")
    
    # Marcar los ecos con líneas verticales y etiquetas
    if echoes:
        for i, echo in enumerate(echoes):
            delay_sec = echo.delay_samples / rx_signal.sample_rate
            ax2.axvline(x=delay_sec, color='green', linestyle='--', alpha=0.7)
            # Colocamos el texto ligeramente a la derecha de la línea
            ax2.text(delay_sec + 0.005, np.max(rx_signal.samples) * 0.7,
                     f'Eco {i+1}\nalpha={echo.attenuation}',
                     color='darkgreen', fontsize=9, ha='left')

    ax2.set_title("Señal Recibida en el Micrófono")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    _save_or_show(fig, filepath)

def plot_chirp_zoom(tx_signal: Signal, zoom_ms: float = 5.0, filepath: str = None):
    """Genera una vista ampliada de los primeros milisegundos del chirp."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    zoom_seconds = zoom_ms / 1000.0
    samples_in_zoom = int(zoom_seconds * tx_signal.sample_rate)
    
    t_zoom = tx_signal.time_vector[:samples_in_zoom] * 1000.0 # Convertir a ms para el eje X
    s_zoom = tx_signal.samples[:samples_in_zoom]
    
    ax.plot(t_zoom, s_zoom, color='blue', marker='.')
    ax.set_title(f"Vista Ampliada del Chirp (0 a {zoom_ms} ms)")
    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Amplitud")
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    _save_or_show(fig, filepath)

def plot_correlation(corr_result: CorrelationResult, peak_lag: int = None, filepath: str = None):
    """Grafica el resultado de la correlación cruzada y marca el pico si se provee."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Graficamos la magnitud absoluta de la correlación
    ax.plot(corr_result.lags, np.abs(corr_result.values), color='purple', label="Magnitud de Correlación")
    
    if peak_lag is not None:
        # Encontrar el valor en el pico para graficar el punto
        peak_idx = np.where(corr_result.lags == peak_lag)[0][0]
        peak_val = np.abs(corr_result.values[peak_idx])
        ax.plot(peak_lag, peak_val, 'ro', markersize=8, label=f"Eco Detectado (lag={peak_lag})")
        ax.axvline(x=peak_lag, color='red', linestyle='--', alpha=0.5)
        
    ax.set_title(f"Correlación Cruzada (Método: {corr_result.method.upper()})")
    ax.set_xlabel("Desplazamiento (Lags / Muestras)")
    ax.set_ylabel("Magnitud")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    _save_or_show(fig, filepath)