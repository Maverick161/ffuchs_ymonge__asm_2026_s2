"""
Experimento: experiment_ranging.py
----------------------------------
Responsabilidad: Validar cuantitativamente la cadena completa de estimación
de distancia del radar acústico y generar las figuras correspondientes
para la documentación del proyecto.

El experimento integra:

Generación del chirp -> Simulación acústica -> Ruido AWGN ->
Correlación FFT -> Detección -> Estimación de distancia.

Implementa:
- Evaluación del error producido por la discretización temporal.
- Validación estadística de distancia bajo ruido.
- Análisis del efecto de la temperatura ambiente.
- Diferenciación entre espaciado por muestra y resolución física ideal.
- Calibración de un umbral de detección mediante escenarios sin eco.
- Estimación de la tasa observada de falsas alarmas P_FA.
- Barrido de SNR para estudiar la tasa observada de detección P_D.
- Uso del mismo umbral normalizado para P_D y P_FA.
- Generación automática de gráficas para los principales resultados.
"""

import os
import numpy as np

from radar.signals.signal import Signal
from radar.signals.generators import generate_chirp
from radar.simulation.echoes import EchoConfig, simulate_echoes
from radar.analysis.correlation import cross_correlate_fft
from radar.analysis.detection import estimate_delay
from radar.analysis.ranging import (
    calculate_speed_of_sound,
    calculate_distance,
    estimate_range,
)
from radar.visualization.plots import (
    plot_ranging_distance_validation,
    plot_quantization_error,
    plot_thermal_compensation,
    plot_detection_probability,
    plot_false_alarm_threshold,
)


# ================================================================
# CONFIGURACIÓN GENERAL
# ================================================================

# Frecuencia inicial del chirp transmitido.
F_START = 2000.0

# Frecuencia final del chirp transmitido.
F_END = 8000.0

# Frecuencia de muestreo del sistema.
FS = 48000.0

# Duración del chirp transmitido.
DURATION = 0.05

# Temperatura real utilizada para simular el canal acústico.
SIMULATION_TEMPERATURE = 20.0

# Varianza base de ruido utilizada en las pruebas de distancia.
NOISE_VARIANCE = 0.05

# Número de realizaciones para validar cada distancia.
DISTANCE_REPETITIONS = 30

# Número de realizaciones para cada punto del barrido de SNR.
SNR_REPETITIONS = 100

# Número de señales de solo ruido utilizadas para calibrar el umbral.
THRESHOLD_CALIBRATION_TRIALS = 500

# Número de señales independientes utilizadas para evaluar P_FA.
FALSE_ALARM_TRIALS = 1000

# Probabilidad de falsa alarma utilizada para calibrar el umbral.
TARGET_PFA = 0.01

# Una detección se considera correctamente localizada si se encuentra
# como máximo a ±1 muestra respecto al retardo real.
DETECTION_TOLERANCE_SAMPLES = 1

# Distancia utilizada para estudiar P_D y P_FA.
DETECTION_TEST_DISTANCE = 15.0

# Atenuación fija utilizada durante el barrido de SNR.
DETECTION_TEST_ATTENUATION = 0.30

# Directorio donde se almacenarán las figuras de este experimento.
FIGURES_DIRECTORY = "results/figures/ranging"


def calculate_echo_snr_db(
    tx_samples: np.ndarray,
    attenuation: float,
    noise_variance: float
) -> float:
    """
    Calcula la SNR aproximada del eco durante la ventana activa del chirp.

    La potencia del eco se obtiene a partir de la potencia promedio
    de la señal transmitida y su factor de atenuación.
    """

    # Calculamos la potencia media de la señal transmitida.
    signal_power = np.mean(
        np.abs(tx_samples) ** 2
    )

    # La potencia del eco se modifica según attenuation^2.
    echo_power = (
        attenuation ** 2
    ) * signal_power

    # En AWGN de media cero, la varianza representa
    # directamente la potencia promedio del ruido.
    noise_power = noise_variance

    # Si no existe ruido, la SNR es teóricamente infinita.
    if noise_power == 0.0:
        return float("inf")

    # Calculamos la relación señal-ruido en decibelios.
    snr_db = 10.0 * np.log10(
        echo_power / noise_power
    )

    # Retornamos el resultado.
    return float(snr_db)


def calculate_noise_variance_for_snr(
    tx_samples: np.ndarray,
    attenuation: float,
    target_snr_db: float
) -> float:
    """
    Calcula la varianza de ruido necesaria para producir
    aproximadamente una SNR objetivo determinada.
    """

    # Calculamos la potencia promedio del chirp.
    signal_power = np.mean(
        np.abs(tx_samples) ** 2
    )

    # Calculamos la potencia correspondiente al eco atenuado.
    echo_power = (
        attenuation ** 2
    ) * signal_power

    # Convertimos la SNR desde decibelios hacia escala lineal.
    snr_linear = 10.0 ** (
        target_snr_db / 10.0
    )

    # Despejamos la potencia del ruido a partir de:
    #
    # SNR = P_eco / P_ruido
    noise_variance = (
        echo_power / snr_linear
    )

    # Retornamos la varianza necesaria.
    return float(noise_variance)


def calculate_error_metrics(
    errors: np.ndarray
) -> tuple:
    """
    Calcula las principales métricas estadísticas del error.

    Retorna:
    - MAE.
    - RMSE.
    - Desviación estándar.
    - Máximo error absoluto.
    """

    # Calculamos el error absoluto medio.
    mae = np.mean(
        np.abs(errors)
    )

    # Calculamos la raíz del error cuadrático medio.
    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # Calculamos la desviación estándar muestral.
    if len(errors) > 1:
        std_error = np.std(
            errors,
            ddof=1
        )
    else:
        std_error = 0.0

    # Calculamos el máximo error absoluto observado.
    max_error = np.max(
        np.abs(errors)
    )

    # Retornamos las métricas.
    return mae, rmse, std_error, max_error


def create_noise_only_signal(
    number_of_samples: int,
    sample_rate: float,
    noise_variance: float,
    rng: np.random.Generator
) -> Signal:
    """
    Genera una señal RX formada únicamente por ruido AWGN.

    Esta señal representa la hipótesis H0:
    no existe ningún eco real.
    """

    # Convertimos la varianza en desviación estándar.
    noise_std = np.sqrt(
        noise_variance
    )

    # Generamos ruido gaussiano blanco de media cero.
    noise_samples = rng.normal(
        0.0,
        noise_std,
        number_of_samples
    )

    # Construimos un objeto Signal compatible con el proyecto.
    noise_signal = Signal(
        samples=noise_samples,
        sample_rate=sample_rate,
        name="Noise_Only_RX"
    )

    # Retornamos la señal.
    return noise_signal


def calculate_normalized_peak_score(
    corr_result,
    tx_norm: float,
    rx_signal: Signal
) -> float:
    """
    Calcula el estadístico normalizado utilizado para detectar ecos.

    score = max(|Rxy[k]|) / (||x|| * ||y||)

    La norma de TX se recibe precalculada para evitar repetir
    innecesariamente el mismo cálculo durante miles de iteraciones.
    """

    # Calculamos únicamente la norma de RX, ya que TX es siempre igual.
    rx_norm = np.linalg.norm(
        rx_signal.samples
    )

    # Construimos el denominador de normalización.
    denominator = (
        tx_norm * rx_norm
    )

    # Evitamos una división inválida en el caso de energía cero.
    if denominator <= 0.0:
        raise ValueError(
            "No se puede normalizar la correlación "
            "de una señal con energía cero."
        )

    # Buscamos el mayor valor absoluto de correlación.
    correlation_peak = np.max(
        np.abs(corr_result.values)
    )

    # Normalizamos el pico respecto a la energía de TX y RX.
    normalized_score = (
        correlation_peak / denominator
    )

    # Retornamos el estadístico.
    return float(normalized_score)


def run_ranging_experiment():
    """
    Ejecuta la validación final del sistema de ranging acústico.

    Analiza:
    1. Error de discretización.
    2. Robustez de distancia bajo ruido.
    3. Error térmico.
    4. Espaciado de rango y resolución ideal.
    5. Calibración del umbral y P_FA.
    6. P_D en función de la SNR.
    7. Generación automática de gráficas.
    """

    # Imprimimos el encabezado general.
    print(
        "--- Experimento Final: Estimación Física de Distancia (Ranging) ---"
    )

    # ================================================================
    # PREPARACIÓN GENERAL
    # ================================================================

    # Generamos el chirp conocido utilizado por el sistema.
    tx_signal = generate_chirp(
        F_START,
        F_END,
        FS,
        DURATION
    )

    # Calculamos una sola vez la norma de TX.
    # Esta cantidad será reutilizada durante todas las pruebas
    # de detección y falsas alarmas.
    tx_norm = np.linalg.norm(
        tx_signal.samples
    )

    # Definimos las distancias utilizadas para validar el ranging.
    real_distances = [
        2.5,
        10.0,
        25.0
    ]

    # Calculamos la velocidad real del sonido correspondiente
    # a la temperatura simulada.
    real_speed = calculate_speed_of_sound(
        SIMULATION_TEMPERATURE
    )

    # Inicializamos el generador pseudoaleatorio con semilla fija.
    rng = np.random.default_rng(
        2026
    )

    # Creamos el directorio de salida de figuras.
    os.makedirs(
        FIGURES_DIRECTORY,
        exist_ok=True
    )

    # ================================================================
    # CONTENEDORES PARA LAS GRÁFICAS
    # ================================================================

    # Distancias reales utilizadas en la validación.
    graph_real_distances = []

    # Distancias promedio estimadas.
    graph_estimated_distances = []

    # Errores de cuantización expresados en milímetros.
    graph_quantization_errors_mm = []

    # Temperaturas evaluadas.
    graph_temperatures = []

    # Distancias obtenidas con el modelo continuo.
    graph_ideal_distances = []

    # Distancias obtenidas con el pipeline digital.
    graph_digital_distances = []

    # Valores de SNR utilizados en el barrido.
    graph_snr_values = []

    # P_D observada para cada SNR.
    graph_pd_values = []

    # ================================================================
    # 1. ERROR INTRODUCIDO POR LA DISCRETIZACIÓN TEMPORAL
    # ================================================================

    print(
        "\n[1] ERROR INTRODUCIDO POR LA DISCRETIZACIÓN TEMPORAL"
    )

    print(
        f"{'D. Real (m)':<13} | "
        f"{'Delay exacto':<14} | "
        f"{'Delay sim.':<11} | "
        f"{'D. representada (m)':<20} | "
        f"{'Error cuant. (mm)':<17}"
    )

    print("-" * 90)

    # Evaluamos individualmente cada distancia.
    for distance in real_distances:

        # Calculamos el tiempo exacto de ida y vuelta.
        exact_delay_seconds = (
            2.0 * distance
        ) / real_speed

        # Convertimos el tiempo continuo a un número real de muestras.
        exact_delay_samples = (
            exact_delay_seconds * FS
        )

        # El simulador utiliza únicamente retardos enteros.
        simulated_delay_samples = int(
            round(exact_delay_samples)
        )

        # Calculamos la distancia que representa realmente
        # ese número entero de muestras.
        represented_distance = estimate_range(
            simulated_delay_samples,
            FS,
            temperature_c=SIMULATION_TEMPERATURE
        )

        # Calculamos el error producido por la discretización.
        quantization_error = (
            represented_distance - distance
        )

        # Convertimos el error a milímetros.
        quantization_error_mm = (
            quantization_error * 1000.0
        )

        # Guardamos el resultado para su gráfica posterior.
        graph_quantization_errors_mm.append(
            quantization_error_mm
        )

        # Imprimimos los resultados.
        print(
            f"{distance:<13.2f} | "
            f"{exact_delay_samples:<14.3f} | "
            f"{simulated_delay_samples:<11d} | "
            f"{represented_distance:<20.6f} | "
            f"{quantization_error_mm:<17.3f}"
        )

    # ================================================================
    # 2. ROBUSTEZ DE LA ESTIMACIÓN BAJO RUIDO
    # ================================================================

    print(
        "\n[2] ROBUSTEZ DE LA ESTIMACIÓN BAJO RUIDO "
        f"({DISTANCE_REPETITIONS} repeticiones por distancia)"
    )

    print(
        f"{'D. Real':<8} | "
        f"{'Lag Real':<8} | "
        f"{'SNR (dB)':<9} | "
        f"{'Lag Med.':<8} | "
        f"{'Det. obs. (%)':<13} | "
        f"{'MAE (m)':<10} | "
        f"{'RMSE (m)':<10} | "
        f"{'STD (m)':<10} | "
        f"{'Max |e| (m)':<12}"
    )

    print("-" * 120)

    # Evaluamos cada distancia de manera independiente.
    for distance in real_distances:

        # Calculamos el retardo correspondiente.
        delay_seconds = (
            2.0 * distance
        ) / real_speed

        # Convertimos el retardo a muestras enteras.
        delay_samples = int(
            round(delay_seconds * FS)
        )

        # Aplicamos el modelo simplificado de reducción
        # de amplitud con la distancia.
        attenuation = max(
            0.05,
            1.0 - (distance / 30.0)
        )

        # Calculamos la SNR aproximada del escenario.
        snr_db = calculate_echo_snr_db(
            tx_signal.samples,
            attenuation,
            NOISE_VARIANCE
        )

        # Calculamos la distancia correspondiente exactamente
        # al lag entero utilizado por el simulador.
        represented_distance = estimate_range(
            delay_samples,
            FS,
            temperature_c=SIMULATION_TEMPERATURE
        )

        # Creamos la configuración del eco una única vez.
        # No cambia durante las repeticiones.
        echo = [
            EchoConfig(
                delay_samples=delay_samples,
                attenuation=attenuation
            )
        ]

        # Inicializamos los contenedores estadísticos.
        detected_lags = []
        lag_errors = []
        distance_errors = []
        estimated_distances = []

        # Contador de detecciones correctamente localizadas.
        correct_detections = 0

        # Ejecutamos múltiples realizaciones independientes.
        for _ in range(DISTANCE_REPETITIONS):

            # Generamos una nueva realización de RX.
            rx_signal = simulate_echoes(
                tx_signal,
                echo,
                noise_variance=NOISE_VARIANCE,
                rng=rng
            )

            # Calculamos la correlación mediante FFT.
            corr_result = cross_correlate_fft(
                tx_signal,
                rx_signal
            )

            # Localizamos el máximo de correlación.
            detected_samples, _ = estimate_delay(
                corr_result
            )

            # Convertimos el lag detectado a distancia.
            estimated_distance = estimate_range(
                detected_samples,
                FS,
                temperature_c=SIMULATION_TEMPERATURE
            )

            # Calculamos el error temporal.
            lag_error = (
                detected_samples - delay_samples
            )

            # Calculamos el error espacial.
            distance_error = (
                estimated_distance - distance
            )

            # Almacenamos los resultados.
            detected_lags.append(
                detected_samples
            )

            lag_errors.append(
                lag_error
            )

            distance_errors.append(
                distance_error
            )

            estimated_distances.append(
                estimated_distance
            )

            # Contabilizamos una localización correcta
            # si el error es como máximo de ±1 muestra.
            if (
                abs(lag_error)
                <= DETECTION_TOLERANCE_SAMPLES
            ):
                correct_detections += 1

        # Convertimos las listas a arreglos NumPy.
        detected_lags = np.asarray(
            detected_lags,
            dtype=float
        )

        lag_errors = np.asarray(
            lag_errors,
            dtype=float
        )

        distance_errors = np.asarray(
            distance_errors,
            dtype=float
        )

        estimated_distances = np.asarray(
            estimated_distances,
            dtype=float
        )

        # Calculamos el lag mediano.
        median_lag = np.median(
            detected_lags
        )

        # Calculamos la tasa observada de detecciones correctas.
        observed_detection_rate = (
            correct_detections
            / DISTANCE_REPETITIONS
        ) * 100.0

        # Calculamos métricas del error espacial.
        mae, rmse, std_error, max_error = calculate_error_metrics(
            distance_errors
        )

        # Calculamos el error exclusivamente producido por cuantización.
        quantization_error = (
            represented_distance - distance
        )

        # Calculamos el error temporal medio del detector.
        mean_detector_error = np.mean(
            np.abs(lag_errors)
        )

        # Calculamos la distancia promedio de las realizaciones.
        mean_estimated_distance = np.mean(
            estimated_distances
        )

        # Guardamos los valores para la gráfica real vs estimada.
        graph_real_distances.append(
            distance
        )

        graph_estimated_distances.append(
            mean_estimated_distance
        )

        # Mostramos los resultados.
        print(
            f"{distance:<8.2f} | "
            f"{delay_samples:<8d} | "
            f"{snr_db:<9.2f} | "
            f"{median_lag:<8.1f} | "
            f"{observed_detection_rate:<13.2f} | "
            f"{mae:<10.6f} | "
            f"{rmse:<10.6f} | "
            f"{std_error:<10.6f} | "
            f"{max_error:<12.6f}"
        )

        print(
            f"         Error de cuantización: "
            f"{quantization_error * 1000.0:.3f} mm | "
            f"Error medio del detector: "
            f"{mean_detector_error:.3f} muestras"
        )

    # ================================================================
    # 3. IMPACTO DE LA COMPENSACIÓN TÉRMICA
    # ================================================================

    print(
        "\n[3] IMPACTO DE LA COMPENSACIÓN TÉRMICA"
    )

    # Fijamos un objeto a 15 metros.
    fixed_distance = 15.0

    # Calculamos el tiempo continuo de vuelo.
    fixed_delay_seconds = (
        2.0 * fixed_distance
    ) / real_speed

    # Calculamos su representación discreta.
    fixed_delay_samples = int(
        round(fixed_delay_seconds * FS)
    )

    # Temperaturas utilizadas en la prueba.
    temperatures = [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0
    ]

    print(
        f"{'Temp (°C)':<10} | "
        f"{'c (m/s)':<10} | "
        f"{'D. ideal (m)':<13} | "
        f"{'Err. térmico (m)':<16} | "
        f"{'D. digital (m)':<14} | "
        f"{'Err. total (m)':<14}"
    )

    print("-" * 90)

    # Evaluamos cada temperatura asumida.
    for temperature in temperatures:

        # Calculamos la velocidad del sonido supuesta.
        assumed_speed = calculate_speed_of_sound(
            temperature
        )

        # Calculamos la distancia continua para aislar
        # únicamente el error térmico.
        ideal_distance = calculate_distance(
            fixed_delay_seconds,
            speed_of_sound=assumed_speed
        )

        # Calculamos el error térmico puro.
        thermal_error = (
            ideal_distance - fixed_distance
        )

        # Calculamos el resultado del pipeline digital.
        digital_distance = estimate_range(
            fixed_delay_samples,
            FS,
            temperature_c=temperature
        )

        # Calculamos el error total.
        total_error = (
            digital_distance - fixed_distance
        )

        # Guardamos los datos necesarios para la gráfica.
        graph_temperatures.append(
            temperature
        )

        graph_ideal_distances.append(
            ideal_distance
        )

        graph_digital_distances.append(
            digital_distance
        )

        # Mostramos los resultados.
        print(
            f"{temperature:<10.1f} | "
            f"{assumed_speed:<10.2f} | "
            f"{ideal_distance:<13.4f} | "
            f"{thermal_error:<16.4f} | "
            f"{digital_distance:<14.4f} | "
            f"{total_error:<14.4f}"
        )

    # ================================================================
    # 4. DISCRETIZACIÓN Y RESOLUCIÓN
    # ================================================================

    print(
        "\n[4] DISCRETIZACIÓN TEMPORAL, ESPACIADO DE RANGO "
        "Y RESOLUCIÓN IDEAL"
    )

    # Calculamos el periodo correspondiente a una muestra.
    sample_period = (
        1.0 / FS
    )

    # Calculamos cuánto cambia la distancia estimada
    # cuando el lag cambia exactamente una muestra.
    range_bin_spacing = (
        real_speed / (2.0 * FS)
    )

    # Al redondear a la muestra más próxima, el máximo
    # error ideal de cuantización es medio bin.
    max_quantization_error = (
        range_bin_spacing / 2.0
    )

    # Calculamos el ancho de banda del chirp.
    chirp_bandwidth = (
        F_END - F_START
    )

    # Calculamos la resolución ideal asociada al ancho de banda.
    ideal_range_resolution = (
        real_speed
        / (2.0 * chirp_bandwidth)
    )

    # Calculamos el producto tiempo-ancho de banda.
    time_bandwidth_product = (
        chirp_bandwidth * DURATION
    )

    # Calculamos una referencia ideal de ganancia de procesamiento.
    processing_gain_db = (
        10.0
        * np.log10(time_bandwidth_product)
    )

    # Mostramos los resultados.
    print(
        f"Periodo de muestreo: "
        f"{sample_period * 1e6:.3f} us"
    )

    print(
        f"Espaciado de rango equivalente a 1 muestra: "
        f"{range_bin_spacing * 1000.0:.3f} mm"
    )

    print(
        f"Error máximo ideal por redondeo: "
        f"{max_quantization_error * 1000.0:.3f} mm"
    )

    print(
        f"Ancho de banda del chirp: "
        f"{chirp_bandwidth / 1000.0:.2f} kHz"
    )

    print(
        f"Resolución ideal aproximada asociada al ancho de banda: "
        f"{ideal_range_resolution * 100.0:.2f} cm"
    )

    print(
        f"Producto tiempo-ancho de banda BT: "
        f"{time_bandwidth_product:.1f}"
    )

    print(
        f"Ganancia de procesamiento ideal de referencia: "
        f"{processing_gain_db:.2f} dB"
    )

    # ================================================================
    # 5. CALIBRACIÓN DEL UMBRAL Y P_FA
    # ================================================================

    print(
        "\n[5] CALIBRACIÓN DEL UMBRAL Y EVALUACIÓN DE P_FA"
    )

    # Calculamos el retardo utilizado durante las pruebas de detección.
    detection_delay_seconds = (
        2.0 * DETECTION_TEST_DISTANCE
    ) / real_speed

    detection_delay_samples = int(
        round(detection_delay_seconds * FS)
    )

    # Calculamos la longitud RX que se utilizará tanto
    # bajo H0 como bajo H1.
    detection_rx_length = (
        tx_signal.number_of_samples
        + detection_delay_samples
    )

    # Elegimos una varianza positiva de referencia.
    calibration_noise_variance = 1.0

    # Inicializamos los estadísticos obtenidos bajo H0.
    calibration_scores = []

    print(
        f"Calibrando con {THRESHOLD_CALIBRATION_TRIALS} "
        f"realizaciones sin eco..."
    )

    # Construimos empíricamente la distribución bajo H0.
    for _ in range(THRESHOLD_CALIBRATION_TRIALS):

        # Generamos únicamente ruido.
        noise_rx = create_noise_only_signal(
            detection_rx_length,
            FS,
            calibration_noise_variance,
            rng
        )

        # Calculamos la correlación contra el chirp conocido.
        noise_corr = cross_correlate_fft(
            tx_signal,
            noise_rx
        )

        # Calculamos el estadístico normalizado.
        score = calculate_normalized_peak_score(
            noise_corr,
            tx_norm,
            noise_rx
        )

        # Guardamos el resultado.
        calibration_scores.append(
            score
        )

    # Convertimos a arreglo NumPy.
    calibration_scores = np.asarray(
        calibration_scores,
        dtype=float
    )

    # Para una P_FA objetivo del 1 %, seleccionamos
    # el percentil 99 de la distribución H0.
    threshold_quantile = (
        1.0 - TARGET_PFA
    )

    # Calculamos el umbral de decisión.
    detection_threshold = np.quantile(
        calibration_scores,
        threshold_quantile
    )

    print(
        f"P_FA objetivo para calibración: "
        f"{TARGET_PFA * 100.0:.2f} %"
    )

    print(
        f"Umbral normalizado obtenido: "
        f"{detection_threshold:.6f}"
    )

    # ------------------------------------------------
    # Evaluación independiente de P_FA
    # ------------------------------------------------

    # Inicializamos el contador de falsas alarmas.
    false_alarms = 0

    print(
        f"Evaluando P_FA con {FALSE_ALARM_TRIALS} "
        f"nuevas realizaciones sin eco..."
    )

    # Utilizamos nuevas realizaciones que no participaron
    # en la calibración del umbral.
    for _ in range(FALSE_ALARM_TRIALS):

        # Generamos otra señal formada exclusivamente por ruido.
        noise_rx = create_noise_only_signal(
            detection_rx_length,
            FS,
            calibration_noise_variance,
            rng
        )

        # Calculamos la correlación.
        noise_corr = cross_correlate_fft(
            tx_signal,
            noise_rx
        )

        # Calculamos el mismo estadístico normalizado.
        score = calculate_normalized_peak_score(
            noise_corr,
            tx_norm,
            noise_rx
        )

        # Cualquier cruce del umbral bajo H0
        # representa una falsa alarma.
        if score > detection_threshold:
            false_alarms += 1

    # Calculamos la P_FA observada.
    observed_pfa = (
        false_alarms
        / FALSE_ALARM_TRIALS
    )

    # Mostramos los resultados.
    print(
        f"Falsas alarmas observadas: "
        f"{false_alarms}/{FALSE_ALARM_TRIALS}"
    )

    print(
        f"P_FA observada: "
        f"{observed_pfa * 100.0:.2f} %"
    )

    # ================================================================
    # 6. P_D EN FUNCIÓN DE LA SNR
    # ================================================================

    print(
        "\n[6] TASA OBSERVADA DE DETECCIÓN EN FUNCIÓN DE LA SNR"
    )

    print(
        f"Mismo umbral utilizado en P_FA y P_D: "
        f"{detection_threshold:.6f}"
    )

    # Utilizamos más puntos en la región donde previamente
    # se observó la transición del detector.
    target_snrs_db = [
        5.0,
        0.0,
        -5.0,
        -10.0,
        -15.0,
        -16.0,
        -17.0,
        -18.0,
        -19.0,
        -20.0,
        -21.0,
        -22.0,
        -23.0,
        -24.0,
        -25.0
    ]

    print(
        f"{'SNR (dB)':<9} | "
        f"{'Var. ruido':<12} | "
        f"{'> umbral':<10} | "
        f"{'Correctas':<10} | "
        f"{'Miss':<7} | "
        f"{'Lag err.':<9} | "
        f"{'P_D obs. (%)':<13}"
    )

    print("-" * 90)

    # Creamos una sola vez la configuración del eco.
    detection_echo = [
        EchoConfig(
            delay_samples=detection_delay_samples,
            attenuation=DETECTION_TEST_ATTENUATION
        )
    ]

    # Evaluamos todos los niveles de SNR.
    for target_snr_db in target_snrs_db:

        # Calculamos la varianza necesaria para lograr
        # aproximadamente la SNR deseada.
        snr_noise_variance = calculate_noise_variance_for_snr(
            tx_signal.samples,
            DETECTION_TEST_ATTENUATION,
            target_snr_db
        )

        # Inicializamos los contadores.
        above_threshold = 0
        correct_detections = 0
        misses = 0
        wrong_lag_detections = 0

        # Ejecutamos realizaciones independientes.
        for _ in range(SNR_REPETITIONS):

            # Simulamos H1: eco conocido + ruido.
            rx_signal = simulate_echoes(
                tx_signal,
                detection_echo,
                noise_variance=snr_noise_variance,
                rng=rng
            )

            # Calculamos la correlación por FFT.
            corr_result = cross_correlate_fft(
                tx_signal,
                rx_signal
            )

            # Calculamos el estadístico de detección.
            score = calculate_normalized_peak_score(
                corr_result,
                tx_norm,
                rx_signal
            )

            # Localizamos el máximo de correlación.
            detected_samples, _ = estimate_delay(
                corr_result
            )

            # Calculamos el error temporal.
            lag_error = (
                detected_samples
                - detection_delay_samples
            )

            # Aplicamos exactamente el mismo umbral
            # utilizado durante la evaluación de P_FA.
            if score > detection_threshold:

                # El sistema declaró que existe un eco.
                above_threshold += 1

                # Verificamos además que el lag sea correcto.
                if (
                    abs(lag_error)
                    <= DETECTION_TOLERANCE_SAMPLES
                ):
                    correct_detections += 1

                # El umbral se superó, pero el eco fue
                # localizado en una posición incorrecta.
                else:
                    wrong_lag_detections += 1

            # Si no se supera el umbral, registramos un miss.
            else:
                misses += 1

        # Calculamos la tasa observada de detecciones correctas.
        observed_pd = (
            correct_detections
            / SNR_REPETITIONS
        )

        # Guardamos los datos necesarios para la gráfica P_D(SNR).
        graph_snr_values.append(
            target_snr_db
        )

        graph_pd_values.append(
            observed_pd * 100.0
        )

        # Mostramos los resultados.
        print(
            f"{target_snr_db:<9.1f} | "
            f"{snr_noise_variance:<12.6f} | "
            f"{above_threshold:<10d} | "
            f"{correct_detections:<10d} | "
            f"{misses:<7d} | "
            f"{wrong_lag_detections:<9d} | "
            f"{observed_pd * 100.0:<13.2f}"
        )

    # ================================================================
    # 7. GENERACIÓN DE GRÁFICAS
    # ================================================================

    print(
        "\n[7] GENERACIÓN DE GRÁFICAS DEL EXPERIMENTO"
    )

    # ------------------------------------------------
    # Figura 1: Distancia real vs estimada
    # ------------------------------------------------

    distance_filepath = os.path.join(
        FIGURES_DIRECTORY,
        "distance_validation.png"
    )

    plot_ranging_distance_validation(
        graph_real_distances,
        graph_estimated_distances,
        filepath=distance_filepath
    )

    print(
        f"Gráfica exportada a: {distance_filepath}"
    )

    # ------------------------------------------------
    # Figura 2: Error de cuantización
    # ------------------------------------------------

    quantization_filepath = os.path.join(
        FIGURES_DIRECTORY,
        "quantization_error.png"
    )

    plot_quantization_error(
        real_distances,
        graph_quantization_errors_mm,
        max_quantization_error_mm=(
            max_quantization_error * 1000.0
        ),
        filepath=quantization_filepath
    )

    print(
        f"Gráfica exportada a: {quantization_filepath}"
    )

    # ------------------------------------------------
    # Figura 3: Compensación térmica
    # ------------------------------------------------

    thermal_filepath = os.path.join(
        FIGURES_DIRECTORY,
        "thermal_compensation.png"
    )

    plot_thermal_compensation(
        graph_temperatures,
        graph_ideal_distances,
        graph_digital_distances,
        real_distance=fixed_distance,
        filepath=thermal_filepath
    )

    print(
        f"Gráfica exportada a: {thermal_filepath}"
    )

    # ------------------------------------------------
    # Figura 4: P_D vs SNR
    # ------------------------------------------------

    detection_filepath = os.path.join(
        FIGURES_DIRECTORY,
        "pd_vs_snr.png"
    )

    plot_detection_probability(
        graph_snr_values,
        graph_pd_values,
        filepath=detection_filepath
    )

    print(
        f"Gráfica exportada a: {detection_filepath}"
    )

    # ------------------------------------------------
    # Figura 5: Distribución H0 + umbral
    # ------------------------------------------------

    false_alarm_filepath = os.path.join(
        FIGURES_DIRECTORY,
        "false_alarm_threshold.png"
    )

    plot_false_alarm_threshold(
        calibration_scores,
        detection_threshold,
        target_pfa=TARGET_PFA,
        observed_pfa=observed_pfa,
        filepath=false_alarm_filepath
    )

    print(
        f"Gráfica exportada a: {false_alarm_filepath}"
    )

    # ================================================================
    # 8. CONCLUSIONES AUTOMÁTICAS
    # ================================================================

    # Calculamos la velocidad correspondiente a una temperatura
    # 10 °C mayor que la real.
    speed_plus_10 = calculate_speed_of_sound(
        SIMULATION_TEMPERATURE + 10.0
    )

    # Calculamos el error relativo de velocidad.
    relative_thermal_error = (
        speed_plus_10 - real_speed
    ) / real_speed

    # Convertimos el error relativo en metros para 10 m.
    thermal_error_per_10m = (
        relative_thermal_error * 10.0
    )

    print(
        "\n[*] CONCLUSIONES AUTOMÁTICAS"
    )

    print(
        f"- A {FS / 1000.0:.0f} kHz, una muestra modifica la "
        f"estimación de distancia aproximadamente "
        f"{range_bin_spacing * 1000.0:.2f} mm."
    )

    print(
        "- Este valor representa el espaciado de rango por muestra, "
        "no la resolución física del radar."
    )

    print(
        f"- Para el chirp de "
        f"{F_START / 1000.0:.0f}-{F_END / 1000.0:.0f} kHz, "
        f"la resolución ideal asociada al ancho de banda es de "
        f"aproximadamente {ideal_range_resolution * 100.0:.2f} cm."
    )

    print(
        f"- Un error de 10 °C alrededor de "
        f"{SIMULATION_TEMPERATURE:.0f} °C produce aproximadamente "
        f"{abs(relative_thermal_error) * 100.0:.2f} % "
        f"de error relativo."
    )

    print(
        f"- Esto equivale aproximadamente a "
        f"{abs(thermal_error_per_10m) * 100.0:.1f} cm "
        f"por cada 10 metros."
    )

    print(
        f"- El umbral se calibró para una P_FA objetivo de "
        f"{TARGET_PFA * 100.0:.2f} % y se utilizó sin cambios "
        f"durante la medición de P_D."
    )

    print(
        f"- P_FA observada con {FALSE_ALARM_TRIALS} pruebas: "
        f"{observed_pfa * 100.0:.2f} %."
    )

    print(
        f"- Cada punto de P_D(SNR) utiliza "
        f"{SNR_REPETITIONS} realizaciones independientes."
    )

    print(
        f"- Se generaron 5 figuras en: "
        f"{FIGURES_DIRECTORY}/"
    )

    print(
        "- Las tasas observadas son estimaciones experimentales "
        "y no garantías absolutas de las probabilidades reales."
    )

    print(
        "- La precisión obtenida corresponde al modelo simulado "
        "y no debe interpretarse todavía como precisión "
        "del hardware físico."
    )

    print(
        "--- Fin del Experimento ---"
    )


if __name__ == "__main__":
    # Ejecutamos el experimento únicamente cuando
    # este archivo se invoca directamente.
    run_ranging_experiment()