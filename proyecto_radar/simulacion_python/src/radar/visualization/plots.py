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

from radar.signals.signal import Signal
from radar.analysis.spectrum import Spectrum

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