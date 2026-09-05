"""
Módulo: test_correlation.py
---------------------------
Responsabilidad: Validar matemáticamente la implementación de correlación
cruzada directa.

Las pruebas verifican:
- Valores completos de correlación.
- Comparación contra numpy.correlate.
- Convención y estructura de los lags.
- Retardos positivos, negativos y nulos.
- Atenuación de señales.
- Autocorrelación.
- Señales de diferentes longitudes.
- Señales complejas y conjugación.
- Señales nulas.
- Conservación de las señales originales.
- Consistencia del objeto CorrelationResult.
- Manejo de frecuencias de muestreo incompatibles.

Convención utilizada:
Un lag positivo significa que la señal recibida RX aparece
después de la señal transmitida TX.

Para ejecutar:
pytest tests/test_correlation.py
"""

import numpy as np
import pytest

from radar.signals.signal import Signal
from radar.analysis.correlation import cross_correlate_direct


# Definimos tolerancias numéricas explícitas para las comparaciones
# entre nuestra implementación y los resultados de referencia.
RTOL = 1e-10
ATOL = 1e-12


def test_correlation_direct_known_delay():
    """
    Caso básico de validación del diseño:

        TX = [1, 2, 3, 4]

        RX = [0, 0, 0, 1, 2, 3, 4, 0]

    RX contiene una copia de TX desplazada exactamente tres muestras
    hacia la derecha.

    De acuerdo con nuestra convención:

        delay positivo -> lag positivo

    Por lo tanto, el máximo de correlación debe aparecer en lag = +3.
    """

    # Construimos la señal transmitida
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_rate=1.0,
        name="TX"
    )

    # Construimos la señal recibida con un retardo conocido de 3 muestras
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

    # Calculamos la correlación utilizando nuestra implementación directa
    result = cross_correlate_direct(tx, rx)

    # Localizamos el índice donde la magnitud de la correlación es máxima
    peak_idx = np.argmax(
        np.abs(result.values)
    )

    # Obtenemos el lag correspondiente al máximo
    peak_lag = result.lags[peak_idx]

    # El retardo conocido debe ser detectado exactamente como +3
    assert peak_lag == 3


def test_correlation_direct_known_complete_result():
    """
    Verifica TODOS los valores de una correlación cuyo resultado
    puede calcularse exactamente.

    Esto evita que una implementación incorrecta pase la prueba
    únicamente porque colocó el máximo en el lugar correcto.
    """

    # Definimos la señal transmitida
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_rate=1.0
    )

    # Definimos una copia retardada de TX
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

    # Ejecutamos nuestra correlación
    result = cross_correlate_direct(tx, rx)

    # Resultado completo esperado según la definición de
    # correlación lineal utilizada en el proyecto.
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

    # Los lags correspondientes deben ir desde:
    #
    # -(Nx - 1) hasta Ny - 1
    expected_lags = np.arange(-3, 8)

    # Verificamos TODOS los valores de correlación
    assert np.allclose(
        result.values,
        expected_values,
        rtol=RTOL,
        atol=ATOL
    )

    # Verificamos TODOS los lags
    assert np.array_equal(
        result.lags,
        expected_lags
    )


@pytest.mark.parametrize(
    "nx, ny",
    [
        (2, 4),
        (4, 8),
        (8, 3),
        (8, 8),
        (16, 16),
        (5, 13),
    ]
)
def test_correlation_direct_vs_numpy(nx, ny):
    """
    Prueba de referencia principal.

    Compara nuestra implementación de correlación directa contra
    numpy.correlate() para diferentes combinaciones de longitudes.

    Se prueban situaciones donde:

        Nx < Ny
        Nx = Ny
        Nx > Ny

    Esto es importante porque en el radar normalmente la señal
    recibida será más larga que la señal transmitida.
    """

    # Creamos un generador pseudoaleatorio con semilla determinista
    # para garantizar que la prueba sea completamente reproducible.
    rng = np.random.default_rng(
        2026 + nx + ny
    )

    # Generamos las muestras de TX
    tx_samples = rng.random(nx)

    # Generamos las muestras de RX
    rx_samples = rng.random(ny)

    # Construimos las señales
    tx = Signal(
        samples=tx_samples,
        sample_rate=1000.0,
        name="TX"
    )

    rx = Signal(
        samples=rx_samples,
        sample_rate=1000.0,
        name="RX"
    )

    # Ejecutamos nuestra implementación
    result = cross_correlate_direct(
        tx,
        rx
    )

    # NumPy se utiliza únicamente como implementación de referencia.
    #
    # El orden rx, tx corresponde a nuestra convención:
    #
    # cross_correlate_direct(TX, RX)
    expected_values = np.correlate(
        rx.samples,
        tx.samples,
        mode="full"
    )

    # Construimos los lags correspondientes a nuestra convención
    expected_lags = np.arange(
        -(nx - 1),
        ny
    )

    # Comparamos todos los valores
    assert np.allclose(
        result.values,
        expected_values,
        rtol=RTOL,
        atol=ATOL
    )

    # Comparamos también toda la estructura de lags
    assert np.array_equal(
        result.lags,
        expected_lags
    )


def test_correlation_direct_zero_delay():
    """
    Verifica la autocorrelación de una señal.

    Cuando una señal se correlaciona consigo misma,
    el máximo debe aparecer en:

        lag = 0
    """

    # Creamos una señal conocida
    tx = Signal(
        samples=np.array([
            1.0,
            0.5,
            0.2
        ]),
        sample_rate=100.0
    )

    # Calculamos su autocorrelación
    result = cross_correlate_direct(
        tx,
        tx
    )

    # Encontramos el máximo
    peak_idx = np.argmax(
        np.abs(result.values)
    )

    # El máximo debe encontrarse en lag cero
    assert result.lags[peak_idx] == 0


def test_correlation_direct_autocorrelation_peak_value():
    """
    Verifica también el VALOR del máximo de autocorrelación.

    Para una señal real:

        r_xx[0] = sum |x[n]|^2

    Es decir, el valor en lag cero corresponde a la energía
    de la señal.
    """

    # Definimos una señal conocida
    samples = np.array([
        1.0,
        0.5,
        0.2
    ])

    tx = Signal(
        samples=samples,
        sample_rate=100.0
    )

    # Calculamos la autocorrelación
    result = cross_correlate_direct(
        tx,
        tx
    )

    # Localizamos explícitamente el índice correspondiente a lag cero
    zero_lag_idx = np.where(
        result.lags == 0
    )[0][0]

    # Calculamos la energía teórica de la señal
    expected_energy = np.sum(
        np.abs(samples) ** 2
    )

    # El valor de autocorrelación en lag cero
    # debe coincidir con la energía
    assert np.isclose(
        result.values[zero_lag_idx],
        expected_energy,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_direct_positive_and_negative_delay():
    """
    Verifica explícitamente nuestra convención de signo.

    Si RX aparece tres muestras después de TX:

        correlate(TX, RX) -> lag +3

    Si intercambiamos el orden:

        correlate(RX, TX) -> lag -3

    Esta prueba protege contra errores de signo en la implementación.
    """

    # Señal original
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

    # TX desplazada tres posiciones
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
        sample_rate=1.0,
        name="RX"
    )

    # Primera dirección:
    # TX utilizada como referencia y RX como señal recibida
    result_positive = cross_correlate_direct(
        tx,
        rx
    )

    # Buscamos el máximo
    positive_peak_idx = np.argmax(
        np.abs(result_positive.values)
    )

    # Debemos obtener +3
    assert (
        result_positive.lags[positive_peak_idx]
        == 3
    )

    # Invertimos el orden de las señales
    result_negative = cross_correlate_direct(
        rx,
        tx
    )

    # Buscamos nuevamente el máximo
    negative_peak_idx = np.argmax(
        np.abs(result_negative.values)
    )

    # El desplazamiento debe cambiar de signo
    assert (
        result_negative.lags[negative_peak_idx]
        == -3
    )


def test_correlation_direct_attenuated_echo():
    """
    Verifica que una atenuación de la señal recibida no altere
    la posición temporal del máximo de correlación.

    Modelo:

        RX[n] = alpha * TX[n-d]

    donde:

        alpha = 0.5
        d = 3
    """

    # Señal transmitida
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0,
            4.0
        ]),
        sample_rate=1000.0
    )

    # Señal recibida atenuada a la mitad y retrasada 3 muestras
    rx = Signal(
        samples=np.array([
            0.0,
            0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0
        ]),
        sample_rate=1000.0
    )

    # Calculamos la correlación
    result = cross_correlate_direct(
        tx,
        rx
    )

    # Localizamos el máximo
    peak_idx = np.argmax(
        np.abs(result.values)
    )

    # La atenuación no debe cambiar el delay detectado
    assert result.lags[peak_idx] == 3

    # Para una copia perfectamente alineada y atenuada:
    #
    # peak = alpha * energía(TX)
    expected_peak = (
        0.5
        * np.sum(np.abs(tx.samples) ** 2)
    )

    # Verificamos también la amplitud del máximo
    assert np.isclose(
        result.values[peak_idx],
        expected_peak,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_direct_lags_structure():
    """
    Verifica que los lags cubran completamente el espacio
    de una correlación lineal en modo 'full'.

    Para:

        Nx = longitud TX
        Ny = longitud RX

    el resultado debe contener:

        Nx + Ny - 1

    valores.

    Los lags deben ir desde:

        -(Nx - 1)

    hasta:

        Ny - 1
    """

    # Definimos tamaños diferentes
    nx = 4
    ny = 8

    # Creamos las señales
    tx = Signal(
        samples=np.ones(nx),
        sample_rate=1.0
    )

    rx = Signal(
        samples=np.ones(ny),
        sample_rate=1.0
    )

    # Calculamos la correlación
    result = cross_correlate_direct(
        tx,
        rx
    )

    # La longitud completa de una correlación lineal es:
    #
    # Nx + Ny - 1
    expected_length = nx + ny - 1

    # Verificamos tamaño de valores
    assert len(result.values) == expected_length

    # Verificamos tamaño de lags
    assert len(result.lags) == expected_length

    # Construimos explícitamente TODOS los lags esperados
    expected_lags = np.arange(
        -(nx - 1),
        ny
    )

    # Verificamos la secuencia completa
    assert np.array_equal(
        result.lags,
        expected_lags
    )


def test_correlation_direct_complex_signal():
    """
    Verifica el funcionamiento con señales complejas.

    Esta prueba es especialmente importante porque la definición
    de correlación cruzada requiere utilizar el conjugado complejo
    de la señal de referencia.

    Una implementación que olvide la conjugación podría funcionar
    correctamente con señales reales pero fallar en este caso.
    """

    # Definimos una señal TX compleja
    tx_samples = np.array([
        1.0 + 2.0j,
        3.0 - 1.0j,
        -2.0 + 0.5j,
        0.5 + 3.0j
    ], dtype=np.complex128)

    # Definimos una señal RX compleja diferente
    rx_samples = np.array([
        0.5 - 1.0j,
        1.0 + 0.25j,
        2.0 - 2.0j,
        -1.0 + 0.5j,
        3.0 + 1.0j
    ], dtype=np.complex128)

    # Construimos ambas señales
    tx = Signal(
        samples=tx_samples,
        sample_rate=1000.0
    )

    rx = Signal(
        samples=rx_samples,
        sample_rate=1000.0
    )

    # Ejecutamos nuestra implementación
    result = cross_correlate_direct(
        tx,
        rx
    )

    # NumPy calcula la correlación de referencia aplicando
    # correctamente conjugación compleja
    expected = np.correlate(
        rx_samples,
        tx_samples,
        mode="full"
    )

    # Nuestra implementación debe ser numéricamente equivalente
    assert np.allclose(
        result.values,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_correlation_direct_zero_received_signal():
    """
    Verifica el comportamiento cuando la señal recibida
    contiene únicamente ceros.

    Matemáticamente:

        y[n] = 0

    implica:

        r_yx[k] = 0

    para cualquier valor de k.
    """

    # Definimos una señal TX no nula
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0
        ]),
        sample_rate=1000.0
    )

    # Definimos una señal RX completamente nula
    rx = Signal(
        samples=np.zeros(8),
        sample_rate=1000.0
    )

    # Calculamos la correlación
    result = cross_correlate_direct(
        tx,
        rx
    )

    # Toda la correlación debe ser cero
    assert np.allclose(
        result.values,
        0.0,
        atol=ATOL
    )


def test_correlation_direct_does_not_modify_signals():
    """
    Verifica una propiedad arquitectónica importante:

    La correlación es una operación de análisis y NO debe
    modificar las señales TX o RX recibidas como entrada.
    """

    # Construimos TX
    tx = Signal(
        samples=np.array([
            1.0,
            2.0,
            3.0
        ]),
        sample_rate=1000.0
    )

    # Construimos RX
    rx = Signal(
        samples=np.array([
            0.0,
            1.0,
            2.0,
            3.0
        ]),
        sample_rate=1000.0
    )

    # Guardamos copias independientes de las muestras
    tx_original = tx.samples.copy()
    rx_original = rx.samples.copy()

    # Ejecutamos la correlación
    cross_correlate_direct(
        tx,
        rx
    )

    # TX debe permanecer exactamente igual
    assert np.array_equal(
        tx.samples,
        tx_original
    )

    # RX debe permanecer exactamente igual
    assert np.array_equal(
        rx.samples,
        rx_original
    )


def test_correlation_direct_result_metadata():
    """
    Verifica que CorrelationResult conserve la información
    necesaria para los módulos posteriores de detección
    y visualización.

    El resultado debe conocer:
    - La frecuencia de muestreo.
    - El método de correlación utilizado.
    """

    # Creamos dos señales compatibles
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

    # Ejecutamos la correlación directa
    result = cross_correlate_direct(
        tx,
        rx
    )

    # La frecuencia de muestreo debe preservarse
    assert result.sample_rate == 48000.0

    # El resultado debe indicar claramente qué método se utilizó.
    #
    # Este test establece como contrato el identificador "direct".
    assert result.method == "direct"


def test_correlation_direct_fails_on_mismatched_sample_rates():
    """
    Verifica que el sistema rechace correlacionar señales
    muestreadas a frecuencias diferentes.

    Un mismo desplazamiento en muestras no representa el mismo
    intervalo temporal cuando:

        Fs_TX != Fs_RX

    Por lo tanto, sería necesario remuestrear antes de correlacionar.
    """

    # TX muestreada a 1000 Hz
    tx = Signal(
        samples=np.array([
            1.0,
            1.0
        ]),
        sample_rate=1000.0
    )

    # RX muestreada a 2000 Hz
    rx = Signal(
        samples=np.array([
            1.0,
            1.0
        ]),
        sample_rate=2000.0
    )

    # La función debe detectar la incompatibilidad
    with pytest.raises(ValueError):
        cross_correlate_direct(
            tx,
            rx
        )