"""
Módulo: test_correlation_fft.py
-------------------------------
Responsabilidad: Validar matemáticamente la implementación de correlación
cruzada basada en FFT.

Las pruebas verifican:
- Detección de retardos conocidos.
- Resultado completo de una correlación conocida.
- Equivalencia entre correlación FFT y correlación directa.
- Comparación independiente contra numpy.correlate.
- Diferentes combinaciones de longitudes Nx y Ny.
- Correcto uso de Zero Padding.
- Prevención de correlación circular accidental.
- Manejo correcto de señales complejas y conjugación.
- Retardos positivos, negativos y nulos.
- Autocorrelación.
- Caso frontera N = 1.
- Señales completamente nulas.
- Estructura completa de los lags.
- Conservación de frecuencia de muestreo y metadatos.
- Inmutabilidad de las señales TX y RX.
- Rechazo de frecuencias de muestreo incompatibles.

Convención utilizada:
Un lag positivo significa que la señal recibida RX aparece
después de la señal transmitida TX.

La correlación FFT debe producir el mismo resultado matemático que:

    cross_correlate_direct()

y que la referencia independiente:

    np.correlate(rx.samples, tx.samples, mode="full")

Para ejecutar:

pytest tests/test_correlation_fft.py
"""

import numpy as np
import pytest

from radar.signals.signal import Signal
from radar.analysis.correlation import (
    cross_correlate_direct,
    cross_correlate_fft,
)


# Definimos tolerancias numéricas explícitas.
#
# La correlación mediante FFT utiliza operaciones en punto flotante
# y puede introducir errores numéricos muy pequeños debido a:
#
# - FFT
# - multiplicación compleja
# - IFFT
#
# Por esta razón no debemos utilizar igualdad exacta para los
# valores de correlación.
RTOL = 1e-10
ATOL = 1e-12


def test_correlation_fft_known_delay():
    """
    Verifica que la implementación mediante FFT detecte
    correctamente un retardo conocido.

    Caso:

        TX = [1, 2, 3, 4]

        RX = [0, 0, 0, 1, 2, 3, 4, 0]

    RX contiene TX desplazada tres muestras hacia la derecha.

    De acuerdo con nuestra convención:

        RX retrasada -> lag positivo

    por lo tanto:

        lag = +3
    """

    # Construimos la señal transmitida
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1.0,
        name="TX"
    )

    # Construimos la señal recibida con un delay conocido
    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0
        ]),
        sample_rate=1.0,
        name="RX"
    )

    # Calculamos la correlación utilizando FFT
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Localizamos el valor de máxima magnitud
    peak_idx = np.argmax(
        np.abs(result.values)
    )

    # Obtenemos su lag correspondiente
    peak_lag = result.lags[peak_idx]

    # El retardo debe ser exactamente +3 muestras
    assert peak_lag == 3


def test_correlation_fft_known_complete_result():
    """
    Verifica TODOS los valores de una correlación cuyo
    resultado se conoce matemáticamente.

    No basta con verificar únicamente que el pico aparezca
    en la posición correcta.

    También queremos comprobar que sean correctos:

    - los valores de correlación,
    - el orden después de la IFFT,
    - el recorte,
    - los lags.
    """

    # Señal transmitida
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1.0
    )

    # Señal recibida retrasada
    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0
        ]),
        sample_rate=1.0
    )

    # Ejecutamos nuestra implementación mediante FFT
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Resultado completo esperado
    expected_values = np.array([
        0.0,
        0.0,
        0.0,
        4.0,
        11.0,
        20.0,
        30.0,
        20.0,
        11.0,
        4.0,
        0.0
    ])

    # Para:
    #
    # Nx = 4
    # Ny = 8
    #
    # los lags deben ir desde:
    #
    # -(Nx - 1) = -3
    #
    # hasta:
    #
    # Ny - 1 = 7
    expected_lags = np.arange(
        -3,
        8
    )

    # Comparamos toda la correlación
    assert np.allclose(
        result.values,
        expected_values,
        rtol=RTOL,
        atol=ATOL
    )

    # Comparamos todos los lags
    assert np.array_equal(
        result.lags,
        expected_lags
    )


@pytest.mark.parametrize(
    "nx, ny",
    [
        # Caso mínimo
        (1, 1),

        # Ambas potencias de dos
        (2, 4),

        # Ambas no potencias de dos
        (3, 5),

        # TX menor que RX
        (5, 13),

        # Tamaños iguales
        (16, 16),

        # Potencias de dos diferentes
        (32, 64),

        # Tamaños arbitrarios que obligan a Zero Padding
        (50, 80),

        # TX mayor que RX
        (80, 50),

        # Caso más grande y no potencia de dos
        (127, 200),
    ]
)
def test_correlation_fft_equivalent_to_direct(nx, ny):
    """
    PRUEBA PRINCIPAL DE EQUIVALENCIA.

    Verifica que la implementación:

        O(N log N)

    mediante FFT produzca el mismo resultado que la implementación:

        O(N²)

    de correlación directa.

    También se prueban diferentes relaciones entre longitudes:

        Nx < Ny
        Nx = Ny
        Nx > Ny

    y tamaños que no son potencias de dos.

    Esto obliga a cross_correlate_fft() a calcular correctamente:

        L_linear = Nx + Ny - 1

    y posteriormente:

        L_fft = next_power_of_two(L_linear)
    """

    # Creamos un generador reproducible.
    #
    # La semilla depende de las dimensiones para generar señales
    # diferentes en cada combinación sin perder reproducibilidad.
    rng = np.random.default_rng(
        2026 + nx * 10 + ny
    )

    # Generamos TX aleatoria
    tx = Signal(
        samples=rng.random(nx),
        sample_rate=48000.0,
        name="TX"
    )

    # Generamos RX aleatoria
    rx = Signal(
        samples=rng.random(ny),
        sample_rate=48000.0,
        name="RX"
    )

    # Correlación directa de referencia interna
    result_direct = cross_correlate_direct(
        tx,
        rx
    )

    # Correlación acelerada mediante FFT
    result_fft = cross_correlate_fft(
        tx,
        rx
    )

    # Ambas implementaciones deben construir exactamente
    # el mismo eje de retardos.
    assert np.array_equal(
        result_fft.lags,
        result_direct.lags
    )

    # Los valores deben ser numéricamente equivalentes.
    #
    # Utilizamos tolerancia porque la ruta FFT introduce
    # pequeños errores de redondeo.
    assert np.allclose(
        result_fft.values,
        result_direct.values,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "nx, ny",
    [
        (2, 3),
        (4, 8),
        (7, 11),
        (16, 16),
        (31, 64),
        (50, 80),
        (80, 50),
    ]
)
def test_correlation_fft_vs_numpy_reference(nx, ny):
    """
    Compara nuestra implementación FFT contra una referencia
    independiente proporcionada por NumPy.

    Esto crea una triangulación de validación:

        FFT propia
            ≈
        directa propia
            ≈
        NumPy

    Si las tres implementaciones coinciden, la confianza en la
    correctitud matemática aumenta significativamente.
    """

    # Generador pseudoaleatorio reproducible
    rng = np.random.default_rng(
        1234 + nx + ny
    )

    # Generamos señales reales
    tx_samples = rng.random(nx)
    rx_samples = rng.random(ny)

    # Construimos TX
    tx = Signal(
        samples=tx_samples,
        sample_rate=1000.0
    )

    # Construimos RX
    rx = Signal(
        samples=rx_samples,
        sample_rate=1000.0
    )

    # Calculamos nuestra correlación FFT
    result_fft = cross_correlate_fft(
        tx,
        rx
    )

    # NumPy funciona como referencia independiente.
    #
    # El orden:
    #
    # np.correlate(RX, TX)
    #
    # coincide con nuestra convención:
    #
    # cross_correlate_fft(TX, RX)
    expected_values = np.correlate(
        rx.samples,
        tx.samples,
        mode="full"
    )

    # Construimos los lags esperados
    expected_lags = np.arange(
        -(nx - 1),
        ny
    )

    # Comparamos todos los valores
    assert np.allclose(
        result_fft.values,
        expected_values,
        rtol=RTOL,
        atol=ATOL
    )

    # Comparamos también el eje completo de lags
    assert np.array_equal(
        result_fft.lags,
        expected_lags
    )


def test_correlation_fft_prevents_circular_correlation():
    """
    PRUEBA CRÍTICA DE ZERO PADDING.

    La multiplicación directa de FFTs sin suficiente Zero Padding
    produce correlación CIRCULAR.

    Nuestro objetivo es obtener correlación LINEAL.

    Para señales de longitud:

        Nx
        Ny

    debe cumplirse:

        L_linear = Nx + Ny - 1

    Esta prueba utiliza señales asimétricas con valores no nulos
    cerca de los extremos para que un posible wrap-around circular
    sea fácil de detectar.
    """

    # Señal TX deliberadamente asimétrica
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # RX contiene información tanto al inicio como al final
    rx = Signal(
        samples=np.array([
            5.0,
            0.0,
            0.0,
            0.0,
            7.0
        ]),
        sample_rate=1000.0
    )

    # Ejecutamos la correlación mediante FFT
    result_fft = cross_correlate_fft(
        tx,
        rx
    )

    # Resultado lineal de referencia
    expected = np.correlate(
        rx.samples,
        tx.samples,
        mode="full"
    )

    # La longitud de una correlación FULL debe ser:
    #
    # Nx + Ny - 1
    expected_length = (
        tx.number_of_samples
        + rx.number_of_samples
        - 1
    )

    # Si accidentalmente obtuviéramos una correlación circular,
    # normalmente la longitud o los valores no coincidirían.
    assert len(result_fft.values) == expected_length

    # Verificamos el resultado completo para detectar cualquier
    # contaminación por wrap-around.
    assert np.allclose(
        result_fft.values,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_fft_complex_signal():
    """
    Verifica el funcionamiento con señales complejas.

    La correlación cruzada requiere conjugación:

        r_yx[k] = sum y[n] * conj(x[n-k])

    En el dominio frecuencial también es necesario respetar
    correctamente esta propiedad.

    Una implementación incorrecta podría pasar todas las pruebas
    con señales reales y fallar únicamente con datos complejos.
    """

    # Señal transmitida compleja
    tx_samples = np.array([
        1.0 + 2.0j,
        3.0 - 1.0j,
        -2.0 + 0.5j,
        0.5 + 3.0j
    ], dtype=np.complex128)

    # Señal recibida compleja
    rx_samples = np.array([
        0.5 - 1.0j,
        1.0 + 0.25j,
        2.0 - 2.0j,
        -1.0 + 0.5j,
        3.0 + 1.0j
    ], dtype=np.complex128)

    # Construimos TX
    tx = Signal(
        samples=tx_samples,
        sample_rate=1000.0
    )

    # Construimos RX
    rx = Signal(
        samples=rx_samples,
        sample_rate=1000.0
    )

    # Calculamos la correlación por FFT
    result_fft = cross_correlate_fft(
        tx,
        rx
    )

    # Calculamos también la correlación directa
    result_direct = cross_correlate_direct(
        tx,
        rx
    )

    # Referencia independiente de NumPy
    expected_numpy = np.correlate(
        rx_samples,
        tx_samples,
        mode="full"
    )

    # FFT y correlación directa deben coincidir
    assert np.allclose(
        result_fft.values,
        result_direct.values,
        rtol=RTOL,
        atol=ATOL
    )

    # FFT también debe coincidir con NumPy
    assert np.allclose(
        result_fft.values,
        expected_numpy,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_fft_positive_and_negative_delay():
    """
    Verifica que el procesamiento mediante FFT conserve
    correctamente la convención de signos.

    Si RX aparece tres muestras después de TX:

        correlate_fft(TX, RX) -> +3

    Si invertimos el orden:

        correlate_fft(RX, TX) -> -3

    Esto permite detectar errores de reordenamiento después
    de aplicar la IFFT.
    """

    # Señal original
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # Señal retardada tres muestras
    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # Caso 1:
    #
    # RX aparece después de TX
    positive_result = cross_correlate_fft(
        tx,
        rx
    )

    # Localizamos el pico
    positive_peak_idx = np.argmax(
        np.abs(positive_result.values)
    )

    # Debe ser +3
    assert (
        positive_result.lags[positive_peak_idx]
        == 3
    )

    # Caso 2:
    #
    # Intercambiamos referencia y señal recibida
    negative_result = cross_correlate_fft(
        rx,
        tx
    )

    # Localizamos el pico
    negative_peak_idx = np.argmax(
        np.abs(negative_result.values)
    )

    # El mismo desplazamiento debe cambiar de signo
    assert (
        negative_result.lags[negative_peak_idx]
        == -3
    )


def test_correlation_fft_autocorrelation():
    """
    Verifica la autocorrelación utilizando FFT.

    Para una señal correlacionada consigo misma:

        r_xx[k]

    el máximo debe encontrarse en:

        lag = 0

    y el valor en ese punto debe corresponder a la energía:

        r_xx[0] = sum |x[n]|²
    """

    # Definimos una señal conocida
    samples = np.array([
        1.0,
        0.5,
        -0.25,
        2.0,
        0.75
    ])

    # Construimos la señal
    tx = Signal(
        samples=samples,
        sample_rate=1000.0
    )

    # Calculamos su autocorrelación mediante FFT
    result = cross_correlate_fft(
        tx,
        tx
    )

    # Localizamos el máximo
    peak_idx = np.argmax(
        np.abs(result.values)
    )

    # Debe aparecer en lag cero
    assert result.lags[peak_idx] == 0

    # Calculamos la energía teórica
    expected_energy = np.sum(
        np.abs(samples) ** 2
    )

    # Localizamos explícitamente el índice correspondiente
    # a lag = 0
    zero_lag_idx = np.where(
        result.lags == 0
    )[0][0]

    # Verificamos el valor del pico
    assert np.isclose(
        result.values[zero_lag_idx],
        expected_energy,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_fft_single_sample():
    """
    Verifica el caso base más pequeño posible.

    Para:

        TX = [2]
        RX = [3]

    la correlación contiene un único valor:

        2 * 3 = 6

    en:

        lag = 0

    Este caso también comprueba que el sistema acepte N = 1,
    recordando que:

        1 = 2^0
    """

    # Señales de una única muestra
    tx = Signal(
        samples=np.array([2.0]),
        sample_rate=1000.0
    )

    rx = Signal(
        samples=np.array([3.0]),
        sample_rate=1000.0
    )

    # Ejecutamos la correlación FFT
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Solo debe existir un valor
    assert len(result.values) == 1

    # Solo debe existir lag cero
    assert np.array_equal(
        result.lags,
        np.array([0])
    )

    # La correlación debe ser exactamente 6,
    # salvo pequeños errores de punto flotante.
    assert np.allclose(
        result.values,
        np.array([6.0]),
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_fft_zero_received_signal():
    """
    Verifica el comportamiento cuando RX contiene únicamente ceros.

    Si:

        RX[n] = 0

    entonces:

        r_yx[k] = 0

    para todos los lags.

    Debido a las operaciones FFT/IFFT podrían existir errores
    numéricos minúsculos, por lo que usamos np.allclose().
    """

    # TX no nula
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # RX completamente nula
    rx = Signal(
        samples=np.zeros(10),
        sample_rate=1000.0
    )

    # Correlación FFT
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Todos los valores deben ser numéricamente cero
    assert np.allclose(
        result.values,
        0.0,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "nx, ny",
    [
        (1, 1),
        (3, 8),
        (8, 3),
        (5, 5),
        (7, 20),
    ]
)
def test_correlation_fft_lags_structure(nx, ny):
    """
    Verifica independientemente la estructura del eje de lags.

    Para una correlación lineal FULL:

        longitud = Nx + Ny - 1

    y los lags deben ser:

        -(Nx - 1), ..., Ny - 1
    """

    # Creamos señales sencillas.
    #
    # Los valores no son importantes para esta prueba;
    # solamente interesa la estructura del resultado.
    tx = Signal(
        samples=np.ones(nx),
        sample_rate=1000.0
    )

    rx = Signal(
        samples=np.ones(ny),
        sample_rate=1000.0
    )

    # Ejecutamos la correlación
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Calculamos la longitud esperada
    expected_length = nx + ny - 1

    # Construimos todos los lags esperados
    expected_lags = np.arange(
        -(nx - 1),
        ny
    )

    # values debe poseer la longitud correcta
    assert len(result.values) == expected_length

    # lags debe poseer la misma longitud
    assert len(result.lags) == expected_length

    # El eje completo debe coincidir
    assert np.array_equal(
        result.lags,
        expected_lags
    )


def test_correlation_fft_result_metadata():
    """
    Verifica que CorrelationResult conserve la información
    necesaria para los módulos posteriores.

    La versión FFT debe indicar:

        method = "fft"

    y conservar la frecuencia de muestreo original.
    """

    # Creamos señales compatibles
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0
        ]),
        sample_rate=48000.0
    )

    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            1.0,
            2.0,
            3.0
        ]),
        sample_rate=48000.0
    )

    # Ejecutamos la correlación mediante FFT
    result = cross_correlate_fft(
        tx,
        rx
    )

    # Fs debe conservarse
    assert result.sample_rate == 48000.0

    # Establecemos como contrato el identificador "fft"
    assert result.method == "fft"


def test_correlation_fft_does_not_modify_signals():
    """
    Verifica que cross_correlate_fft() no modifique las
    señales utilizadas como entrada.

    Esto es especialmente importante porque internamente
    el algoritmo puede realizar:

    - Zero Padding.
    - Conversión a arrays complejos.
    - Reverse.
    - Conjugación.
    - FFT.

    Todas esas operaciones deben realizarse sobre copias
    o nuevos arreglos.
    """

    # Construimos TX
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # Construimos RX
    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # Guardamos copias independientes antes de ejecutar
    # la correlación.
    tx_original = tx.samples.copy()
    rx_original = rx.samples.copy()

    # Ejecutamos la función bajo prueba
    cross_correlate_fft(
        tx,
        rx
    )

    # TX debe permanecer exactamente igual
    assert np.array_equal(
        tx.samples,
        tx_original
    )

    # RX también debe permanecer exactamente igual
    assert np.array_equal(
        rx.samples,
        rx_original
    )


def test_correlation_fft_fails_on_mismatched_sample_rates():
    """
    Verifica que cross_correlate_fft() rechace señales con
    diferentes frecuencias de muestreo.

    Si:

        Fs_TX != Fs_RX

    un mismo lag en muestras no representa el mismo intervalo
    temporal para ambas señales.

    Por lo tanto, deberían remuestrearse antes de realizar
    la correlación.
    """

    # TX a 1000 Hz
    tx = Signal(
        samples=np.array([
            1.0,
            1.0
        ]),
        sample_rate=1000.0
    )

    # RX a 2000 Hz
    rx = Signal(
        samples=np.array([
            1.0,
            1.0
        ]),
        sample_rate=2000.0
    )

    # El sistema debe detectar la incompatibilidad
    with pytest.raises(ValueError):
        cross_correlate_fft(
            tx,
            rx
        )