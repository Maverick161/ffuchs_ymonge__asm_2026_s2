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

def plot_ranging_distance_validation(
    real_distances: list[float],
    estimated_distances: list[float],
    filepath: str = None
):
    """
    Grafica la distancia real frente a la distancia estimada.

    Se incluye la recta ideal y = x como referencia para observar
    visualmente la precisión del sistema de ranging.
    """

    # Convertimos los datos a arreglos NumPy para facilitar los cálculos.
    real_distances = np.asarray(real_distances, dtype=float)
    estimated_distances = np.asarray(estimated_distances, dtype=float)

    # Verificamos que ambas listas tengan la misma cantidad de elementos.
    if real_distances.size != estimated_distances.size:
        raise ValueError(
            "real_distances y estimated_distances deben tener la misma longitud."
        )

    # Creamos la figura.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Graficamos los valores obtenidos experimentalmente.
    ax.plot(
        real_distances,
        estimated_distances,
        'bo-',
        label="Distancia estimada"
    )

    # Obtenemos los extremos para construir la recta ideal y = x.
    min_distance = min(
        np.min(real_distances),
        np.min(estimated_distances)
    )

    max_distance = max(
        np.max(real_distances),
        np.max(estimated_distances)
    )

    # Graficamos la respuesta ideal.
    ax.plot(
        [min_distance, max_distance],
        [min_distance, max_distance],
        'k--',
        label="Estimación ideal"
    )

    # Añadimos el error correspondiente a cada punto en milímetros.
    for real_distance, estimated_distance in zip(
        real_distances,
        estimated_distances
    ):
        error_mm = (
            estimated_distance - real_distance
        ) * 1000.0

        ax.annotate(
            f"{error_mm:+.2f} mm",
            (real_distance, estimated_distance),
            textcoords="offset points",
            xytext=(5, 7),
            fontsize=8
        )

    # Configuramos título y etiquetas.
    ax.set_title("Validación de la Estimación de Distancia")
    ax.set_xlabel("Distancia Real (m)")
    ax.set_ylabel("Distancia Estimada (m)")

    # Activamos cuadrícula y leyenda.
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ajustamos los márgenes.
    plt.tight_layout()

    # Guardamos o mostramos la figura.
    _save_or_show(fig, filepath)


def plot_quantization_error(
    real_distances: list[float],
    quantization_errors_mm: list[float],
    max_quantization_error_mm: float = None,
    filepath: str = None
):
    """
    Grafica el error introducido por la cuantización temporal.

    Permite observar cómo el redondeo del retardo a muestras enteras
    modifica ligeramente la distancia representada.
    """

    # Convertimos los datos a arreglos NumPy.
    real_distances = np.asarray(real_distances, dtype=float)
    quantization_errors_mm = np.asarray(
        quantization_errors_mm,
        dtype=float
    )

    # Verificamos que exista un error para cada distancia.
    if real_distances.size != quantization_errors_mm.size:
        raise ValueError(
            "real_distances y quantization_errors_mm deben tener la misma longitud."
        )

    # Creamos la figura.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Graficamos el error de cuantización para cada distancia.
    ax.plot(
        real_distances,
        quantization_errors_mm,
        'o-',
        label="Error de cuantización"
    )

    # Marcamos el nivel de error cero como referencia.
    ax.axhline(
        y=0.0,
        color='black',
        linestyle='-',
        linewidth=1.0
    )

    # Si se conoce el máximo teórico por redondeo,
    # mostramos sus límites positivo y negativo.
    if max_quantization_error_mm is not None:
        ax.axhline(
            y=max_quantization_error_mm,
            color='red',
            linestyle='--',
            alpha=0.7,
            label="Límite ideal por redondeo"
        )

        ax.axhline(
            y=-max_quantization_error_mm,
            color='red',
            linestyle='--',
            alpha=0.7
        )

    # Añadimos el valor numérico sobre cada punto.
    for distance, error in zip(
        real_distances,
        quantization_errors_mm
    ):
        ax.annotate(
            f"{error:+.3f} mm",
            (distance, error),
            textcoords="offset points",
            xytext=(5, 7),
            fontsize=8
        )

    # Configuramos título y etiquetas.
    ax.set_title("Error de Cuantización en la Estimación de Distancia")
    ax.set_xlabel("Distancia Real (m)")
    ax.set_ylabel("Error de Cuantización (mm)")

    # Activamos cuadrícula y leyenda.
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ajustamos los márgenes.
    plt.tight_layout()

    # Guardamos o mostramos.
    _save_or_show(fig, filepath)


def plot_thermal_compensation(
    temperatures: list[float],
    ideal_distances: list[float],
    digital_distances: list[float],
    real_distance: float,
    filepath: str = None
):
    """
    Grafica el impacto de la temperatura asumida sobre la distancia calculada.

    Compara:
    - La distancia obtenida utilizando el modelo continuo.
    - La distancia obtenida por el pipeline digital.
    - La distancia física real del objeto.
    """

    # Convertimos los datos a arreglos NumPy.
    temperatures = np.asarray(temperatures, dtype=float)
    ideal_distances = np.asarray(ideal_distances, dtype=float)
    digital_distances = np.asarray(digital_distances, dtype=float)

    # Verificamos que todas las series tengan la misma longitud.
    if not (
        temperatures.size
        == ideal_distances.size
        == digital_distances.size
    ):
        raise ValueError(
            "temperatures, ideal_distances y digital_distances "
            "deben tener la misma longitud."
        )

    # Creamos la figura.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Graficamos el resultado continuo ideal.
    ax.plot(
        temperatures,
        ideal_distances,
        'bo-',
        label="Modelo continuo"
    )

    # Graficamos el resultado producido por el sistema digital.
    ax.plot(
        temperatures,
        digital_distances,
        'rs--',
        label="Pipeline digital"
    )

    # Marcamos la distancia física verdadera.
    ax.axhline(
        y=real_distance,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f"Distancia real = {real_distance:.2f} m"
    )

    # Configuramos título y etiquetas.
    ax.set_title("Impacto de la Temperatura en la Estimación de Distancia")
    ax.set_xlabel("Temperatura Asumida (°C)")
    ax.set_ylabel("Distancia Calculada (m)")

    # Activamos cuadrícula y leyenda.
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ajustamos los márgenes.
    plt.tight_layout()

    # Guardamos o mostramos.
    _save_or_show(fig, filepath)


def plot_detection_probability(
    snr_values_db: list[float],
    pd_values_percent: list[float],
    filepath: str = None
):
    """
    Grafica la tasa observada de detección P_D en función de la SNR.

    Permite visualizar la región donde el detector comienza a degradarse
    debido al incremento del ruido.
    """

    # Convertimos los datos a arreglos NumPy.
    snr_values_db = np.asarray(
        snr_values_db,
        dtype=float
    )

    pd_values_percent = np.asarray(
        pd_values_percent,
        dtype=float
    )

    # Verificamos que exista una probabilidad para cada valor de SNR.
    if snr_values_db.size != pd_values_percent.size:
        raise ValueError(
            "snr_values_db y pd_values_percent deben tener la misma longitud."
        )

    # Ordenamos los puntos por SNR para que el eje X avance
    # naturalmente desde las SNR más bajas hasta las más altas.
    order = np.argsort(snr_values_db)

    snr_sorted = snr_values_db[order]
    pd_sorted = pd_values_percent[order]

    # Creamos la figura.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Graficamos la tasa observada de detección.
    ax.plot(
        snr_sorted,
        pd_sorted,
        'bo-',
        label=r"$P_D$ observada"
    )

    # Dibujamos una referencia horizontal del 50 %.
    # Esta línea ayuda a visualizar aproximadamente la región
    # de transición del detector.
    ax.axhline(
        y=50.0,
        color='gray',
        linestyle='--',
        alpha=0.7,
        label=r"Referencia $P_D = 50\%$"
    )

    # Limitamos la escala de probabilidad entre 0 y 100 %.
    ax.set_ylim(-5, 105)

    # Configuramos título y etiquetas.
    ax.set_title("Tasa de Detección en Función de la SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$P_D$ Observada (%)")

    # Activamos cuadrícula.
    ax.grid(True, alpha=0.3)

    # Añadimos la leyenda.
    ax.legend()

    # Ajustamos los márgenes.
    plt.tight_layout()

    # Guardamos o mostramos.
    _save_or_show(fig, filepath)


def plot_false_alarm_threshold(
    calibration_scores: list[float],
    detection_threshold: float,
    target_pfa: float,
    observed_pfa: float = None,
    filepath: str = None
):
    """
    Grafica la distribución del estadístico de detección bajo H0.

    H0 representa el escenario donde únicamente existe ruido y no hay eco.
    La gráfica permite visualizar el umbral utilizado para controlar P_FA.
    """

    # Convertimos los scores a un arreglo NumPy.
    calibration_scores = np.asarray(
        calibration_scores,
        dtype=float
    )

    # Verificamos que existan datos para graficar.
    if calibration_scores.size == 0:
        raise ValueError(
            "calibration_scores no puede estar vacío."
        )

    # Creamos la figura.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Graficamos el histograma de scores obtenidos bajo H0.
    ax.hist(
        calibration_scores,
        bins=30,
        alpha=0.75,
        edgecolor='black',
        label="H0: solo ruido"
    )

    # Marcamos el umbral de decisión.
    ax.axvline(
        x=detection_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=(
            f"Umbral = {detection_threshold:.4f}"
        )
    )

    # Construimos el texto asociado a la P_FA objetivo.
    annotation = (
        f"$P_{{FA}}$ objetivo = {target_pfa * 100.0:.2f}%"
    )

    # Si también se proporciona la P_FA observada,
    # agregamos este valor al texto informativo.
    if observed_pfa is not None:
        annotation += (
            f"\n$P_{{FA}}$ observada = "
            f"{observed_pfa * 100.0:.2f}%"
        )

    # Colocamos la información en la parte superior izquierda.
    ax.text(
        0.03,
        0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.8
        )
    )

    # Configuramos título y etiquetas.
    ax.set_title("Distribución del Estadístico de Detección bajo H0")
    ax.set_xlabel("Score Normalizado de Correlación")
    ax.set_ylabel("Frecuencia")

    # Activamos cuadrícula.
    ax.grid(True, alpha=0.3)

    # Mostramos la leyenda.
    ax.legend()

    # Ajustamos márgenes.
    plt.tight_layout()

    # Guardamos o mostramos.
    _save_or_show(fig, filepath)