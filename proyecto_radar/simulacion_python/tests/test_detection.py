"""
Módulo: test_detection.py
-------------------------
Responsabilidad: Validar la interpretación de los resultados de correlación.

Las pruebas verifican:
- Detección del pico de máxima magnitud.
- Manejo correcto de valores positivos, negativos y complejos.
- Detección de picos ubicados en los extremos.
- Comportamiento determinista cuando existen máximos empatados.
- Rechazo de correlaciones vacías o completamente nulas.
- Conversión correcta de retardos en muestras a segundos.
- Manejo de retardos positivos, negativos y nulos.
- Funcionamiento con diferentes frecuencias de muestreo.
- Rechazo de frecuencias de muestreo inválidas.
- Consistencia entre los arreglos de valores y lags.
- Inmutabilidad del resultado de correlación.

Convención utilizada:
Un lag positivo significa que la señal recibida RX aparece
después de la señal transmitida TX.

Para ejecutar:
pytest tests/test_detection.py
"""

import numpy as np
import pytest

from radar.analysis.correlation import CorrelationResult
from radar.analysis.detection import find_peak, estimate_delay


# Definimos tolerancias numéricas explícitas para las comparaciones
# de valores en punto flotante.
RTOL = 1e-10
ATOL = 1e-12


def test_find_peak_locates_maximum_absolute_value():
    """
    Verifica que find_peak() utilice la MAGNITUD de la correlación
    y no simplemente el valor numérico más grande.

    En este caso existe un pico negativo de -10 en lag = 1.

    Aunque:

        -10 < 3

    su magnitud es:

        |-10| = 10

    y por lo tanto representa el máximo pico de correlación.
    """

    # Definimos los lags desde -2 hasta +2
    lags = np.array([
        -2,
        -1,
        0,
        1,
        2
    ])

    # Creamos un máximo absoluto negativo en lag = 1
    values = np.array([
        0.5,
        2.0,
        -1.0,
        -10.0,
        3.0
    ])

    # Construimos un resultado de correlación artificial
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # Buscamos el lag correspondiente al máximo
    detected_lag = find_peak(
        corr_result
    )

    # El mayor valor absoluto es |-10| = 10,
    # ubicado exactamente en lag = 1
    assert detected_lag == 1


def test_find_peak_with_complex_values():
    """
    Verifica que find_peak() funcione correctamente cuando
    los valores de correlación son complejos.

    En una correlación compleja debe utilizarse:

        |r[k]|

    para determinar la intensidad del pico.

    Una implementación que intente comparar directamente los
    números complejos sería matemáticamente incorrecta.
    """

    # Definimos diferentes posiciones de lag
    lags = np.array([
        -2,
        -1,
        0,
        1
    ])

    # El valor 1-10j posee la mayor magnitud:
    #
    # |1-10j| = sqrt(1^2 + 10^2)
    #         = sqrt(101)
    #
    # y está ubicado en lag = 0.
    values = np.array([
        1.0 + 1.0j,
        2.0 + 2.0j,
        1.0 - 10.0j,
        3.0 + 0.0j
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # Detectamos el pico
    detected_lag = find_peak(
        corr_result
    )

    # La máxima magnitud debe encontrarse en lag cero
    assert detected_lag == 0


def test_find_peak_at_first_position():
    """
    Verifica que el algoritmo pueda detectar correctamente un pico
    ubicado en el primer elemento del arreglo.

    Esta prueba protege contra errores de límites o indexación.
    """

    # El máximo está deliberadamente colocado al inicio
    lags = np.array([
        -3,
        -2,
        -1,
        0
    ])

    values = np.array([
        20.0,
        2.0,
        1.0,
        0.5
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # Buscamos el máximo
    detected_lag = find_peak(
        corr_result
    )

    # Debe detectarse el primer lag
    assert detected_lag == -3


def test_find_peak_at_last_position():
    """
    Verifica que el algoritmo pueda detectar correctamente un pico
    ubicado en el último elemento del arreglo.

    Complementa el caso de borde anterior.
    """

    # El máximo está deliberadamente colocado al final
    lags = np.array([
        0,
        1,
        2,
        3
    ])

    values = np.array([
        0.5,
        1.0,
        2.0,
        20.0
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # Detectamos el máximo
    detected_lag = find_peak(
        corr_result
    )

    # El último lag debe ser seleccionado
    assert detected_lag == 3


def test_find_peak_returns_first_when_maximum_is_tied():
    """
    Define el comportamiento determinista cuando existen dos
    valores con exactamente la misma magnitud máxima.

    Para la versión básica de find_peak(), adoptamos el contrato:

        Si existen varios máximos iguales,
        se devuelve el primero según el orden de los lags.

    La detección de múltiples ecos será responsabilidad posterior
    de una función como detect_echoes().
    """

    # Definimos los lags
    lags = np.array([
        -2,
        -1,
        0,
        1,
        2
    ])

    # Existen dos máximos absolutos iguales:
    #
    # |10|  = 10
    # |-10| = 10
    #
    # El primero aparece en lag = -1.
    values = np.array([
        1.0,
        10.0,
        3.0,
        -10.0,
        2.0
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # Ejecutamos la detección
    detected_lag = find_peak(
        corr_result
    )

    # Por contrato se debe devolver el primer máximo
    assert detected_lag == -1


def test_find_peak_fails_on_empty_result():
    """
    Verifica que no se intente detectar un pico cuando
    la correlación no contiene ningún dato.

    Un resultado vacío no posee información suficiente
    para estimar un retardo.
    """

    # La excepción podría ser generada por CorrelationResult
    # o por find_peak(), dependiendo de dónde se encuentre
    # implementada la validación estructural.
    with pytest.raises(ValueError):

        # Construimos deliberadamente un resultado vacío
        corr_result = CorrelationResult(
            values=np.array([]),
            lags=np.array([]),
            sample_rate=1000.0,
            method="test"
        )

        # Intentamos encontrar un pico inexistente
        find_peak(
            corr_result
        )


def test_find_peak_fails_on_zero_correlation():
    """
    Verifica que una correlación completamente nula no produzca
    una detección falsa.

    Si:

        r[k] = 0

    para todos los lags, entonces no existe ningún pico que
    indique la presencia de la señal conocida.

    np.argmax() devolvería artificialmente el primer índice,
    pero eso NO debe interpretarse como un eco real.
    """

    # Definimos diferentes posiciones de lag
    lags = np.array([
        -2,
        -1,
        0,
        1,
        2
    ])

    # Toda la correlación es exactamente cero
    values = np.zeros(5)

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=1000.0,
        method="test"
    )

    # No existe un pico físicamente interpretable,
    # por lo tanto esperamos una excepción.
    with pytest.raises(ValueError):
        find_peak(
            corr_result
        )


def test_find_peak_fails_on_mismatched_values_and_lags():
    """
    Verifica que exista correspondencia uno a uno entre
    los valores de correlación y sus lags.

    Debe cumplirse siempre:

        len(values) == len(lags)

    Cada valor de correlación necesita exactamente un
    desplazamiento asociado.
    """

    # La validación puede realizarse dentro de CorrelationResult
    # o dentro de find_peak().
    with pytest.raises(ValueError):

        # Creamos deliberadamente tres valores
        # pero solamente dos lags.
        corr_result = CorrelationResult(
            values=np.array([
                1.0,
                5.0,
                2.0
            ]),
            lags=np.array([
                -1,
                0
            ]),
            sample_rate=1000.0,
            method="test"
        )

        # Si el constructor no detecta el problema,
        # find_peak() deberá hacerlo.
        find_peak(
            corr_result
        )


def test_find_peak_does_not_modify_correlation_result():
    """
    Verifica que find_peak() sea una operación únicamente
    de análisis.

    La función no debe modificar:
    - values
    - lags
    """

    # Creamos datos conocidos
    values = np.array([
        1.0,
        2.0,
        10.0,
        3.0
    ])

    lags = np.array([
        -1,
        0,
        1,
        2
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values.copy(),
        lags=lags.copy(),
        sample_rate=1000.0,
        method="test"
    )

    # Guardamos copias independientes antes de ejecutar la función
    original_values = corr_result.values.copy()
    original_lags = corr_result.lags.copy()

    # Ejecutamos la detección
    find_peak(
        corr_result
    )

    # Los valores deben permanecer exactamente iguales
    assert np.array_equal(
        corr_result.values,
        original_values
    )

    # Los lags también deben permanecer intactos
    assert np.array_equal(
        corr_result.lags,
        original_lags
    )


def test_estimate_delay_calculates_seconds_correctly():
    """
    Verifica la conversión matemática fundamental:

        delay_seconds = delay_samples / Fs

    Se utiliza:

        delay_samples = 250
        Fs = 1000 Hz

    por lo tanto:

        delay_seconds = 0.25 s
    """

    # Definimos la frecuencia de muestreo
    fs = 1000.0

    # Simulamos diferentes posiciones posibles
    lags = np.array([
        0,
        100,
        250,
        400
    ])

    # Colocamos el máximo en lag = 250
    values = np.array([
        1.0,
        5.0,
        100.0,
        2.0
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=fs,
        method="test"
    )

    # Estimamos el retardo
    delay_samples, delay_seconds = estimate_delay(
        corr_result
    )

    # Debe detectarse el lag correcto
    assert delay_samples == 250

    # 250 muestras / 1000 muestras por segundo = 0.25 segundos
    assert np.isclose(
        delay_seconds,
        0.25,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "delay_samples, sample_rate, expected_seconds",
    [
        # 100 / 1000 = 0.1 s
        (100, 1000.0, 0.1),

        # 480 / 48000 = 0.01 s
        (480, 48000.0, 0.01),

        # 441 / 44100 = 0.01 s
        (441, 44100.0, 0.01),

        # 2400 / 48000 = 0.05 s
        (2400, 48000.0, 0.05),

        # Caso frontera: delay cero
        (0, 48000.0, 0.0),

        # También verificamos un retardo negativo
        (-200, 1000.0, -0.2),
    ]
)
def test_estimate_delay_for_different_sample_rates(
    delay_samples,
    sample_rate,
    expected_seconds
):
    """
    Verifica la conversión samples -> segundos para diferentes
    frecuencias de muestreo y diferentes retardos.

    Esto evita que la implementación dependa accidentalmente
    de una frecuencia de muestreo específica.
    """

    # Creamos tres lags, colocando el lag bajo prueba en el centro.
    #
    # Los otros valores se construyen de forma que sean diferentes
    # y no interfieran con la detección.
    lags = np.array([
        delay_samples - 1,
        delay_samples,
        delay_samples + 1
    ])

    # El máximo de correlación se coloca deliberadamente
    # en la posición central.
    values = np.array([
        1.0,
        100.0,
        2.0
    ])

    # Construimos el resultado
    corr_result = CorrelationResult(
        values=values,
        lags=lags,
        sample_rate=sample_rate,
        method="test"
    )

    # Ejecutamos la estimación
    detected_samples, detected_seconds = estimate_delay(
        corr_result
    )

    # El delay en muestras debe conservarse exactamente
    assert detected_samples == delay_samples

    # La conversión temporal debe coincidir con:
    #
    # tau = delay_samples / Fs
    assert np.isclose(
        detected_seconds,
        expected_seconds,
        rtol=RTOL,
        atol=ATOL
    )


def test_estimate_delay_zero_delay():
    """
    Verifica explícitamente el caso frontera donde
    la señal no presenta desplazamiento temporal.

    Debe obtenerse:

        delay_samples = 0
        delay_seconds = 0
    """

    # Colocamos el máximo en lag cero
    corr_result = CorrelationResult(
        values=np.array([
            1.0,
            10.0,
            2.0
        ]),
        lags=np.array([
            -1,
            0,
            1
        ]),
        sample_rate=1000.0,
        method="test"
    )

    # Ejecutamos la estimación
    delay_samples, delay_seconds = estimate_delay(
        corr_result
    )

    # No existe desplazamiento en muestras
    assert delay_samples == 0

    # Tampoco debe existir desplazamiento temporal
    assert np.isclose(
        delay_seconds,
        0.0,
        atol=ATOL
    )


def test_estimate_delay_negative_delay():
    """
    Verifica que estimate_delay() preserve correctamente
    el signo de un lag negativo.

    Aunque un eco físico normal se espera después de TX,
    matemáticamente la función debe interpretar correctamente
    cualquier CorrelationResult válido.

    Para:

        lag = -200
        Fs = 1000 Hz

    debe obtenerse:

        tau = -0.2 segundos
    """

    # El máximo se encuentra en lag = -200
    corr_result = CorrelationResult(
        values=np.array([
            20.0,
            1.0,
            2.0
        ]),
        lags=np.array([
            -200,
            0,
            100
        ]),
        sample_rate=1000.0,
        method="test"
    )

    # Calculamos el retardo
    delay_samples, delay_seconds = estimate_delay(
        corr_result
    )

    # El signo del lag debe mantenerse
    assert delay_samples == -200

    # -200 / 1000 = -0.2 segundos
    assert np.isclose(
        delay_seconds,
        -0.2,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "invalid_sample_rate",
    [
        0.0,
        -1.0,
        -1000.0,
        np.nan,
        np.inf,
        -np.inf,
    ]
)
def test_estimate_delay_fails_on_invalid_sample_rate(
    invalid_sample_rate
):
    """
    Verifica que una frecuencia de muestreo inválida sea rechazada.

    Debe cumplirse:

        Fs > 0

    y además Fs debe ser un número finito.

    Por lo tanto se rechazan:
    - cero,
    - valores negativos,
    - NaN,
    - infinito positivo,
    - infinito negativo.
    """

    # La validación puede encontrarse en CorrelationResult
    # o directamente en estimate_delay().
    #
    # Por esta razón colocamos ambas operaciones dentro
    # del contexto pytest.raises().
    with pytest.raises(ValueError):

        # Construimos un resultado con Fs deliberadamente inválida
        corr_result = CorrelationResult(
            values=np.array([
                1.0,
                10.0,
                1.0
            ]),
            lags=np.array([
                -1,
                0,
                1
            ]),
            sample_rate=invalid_sample_rate,
            method="test"
        )

        # Si CorrelationResult permite temporalmente la construcción,
        # estimate_delay() debe detectar la frecuencia inválida.
        estimate_delay(
            corr_result
        )