"""
Módulo: test_ranging.py
-----------------------
Responsabilidad:
Validar matemáticamente y numéricamente las funciones encargadas
de convertir retardos temporales en distancias físicas.

Se prueban:
- Velocidad del sonido según temperatura.
- Tiempo de vuelo -> distancia.
- Conversión muestras -> segundos -> metros.
- Casos frontera.
- Valores físicamente inválidos.
- NaN e infinito.
- Caso utilizado en el experimento real del proyecto.
"""

import math
import pytest

from radar.analysis.ranging import (
    calculate_speed_of_sound,
    calculate_distance,
    estimate_range,
)


# Tolerancias para comparaciones en punto flotante.
RTOL = 1e-9
ATOL = 1e-9


# ============================================================
# VELOCIDAD DEL SONIDO
# ============================================================

def test_calculate_speed_of_sound_standard_temps():
    """
    Verifica valores conocidos obtenidos mediante:

        c(T) = 331.3 + 0.606*T
    """

    assert calculate_speed_of_sound(0.0) == pytest.approx(
        331.3,
        rel=RTOL,
        abs=ATOL
    )

    assert calculate_speed_of_sound(20.0) == pytest.approx(
        343.42,
        rel=RTOL,
        abs=ATOL
    )

    assert calculate_speed_of_sound(25.0) == pytest.approx(
        346.45,
        rel=RTOL,
        abs=ATOL
    )


def test_calculate_speed_of_sound_fails_at_absolute_zero():
    """
    El cero absoluto no constituye una temperatura válida
    para este modelo acústico.
    """

    with pytest.raises(ValueError):
        calculate_speed_of_sound(-273.15)


def test_calculate_speed_of_sound_fails_below_absolute_zero():
    """
    Temperaturas inferiores al cero absoluto deben rechazarse.
    """

    with pytest.raises(ValueError):
        calculate_speed_of_sound(-300.0)


@pytest.mark.parametrize(
    "invalid_temperature",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calculate_speed_of_sound_fails_on_non_finite_temperature(
    invalid_temperature
):
    """
    NaN e infinito no representan temperaturas físicas válidas.
    """

    with pytest.raises(ValueError):
        calculate_speed_of_sound(invalid_temperature)


# ============================================================
# DISTANCIA
# ============================================================

def test_calculate_distance_known_delay():
    """
    Valida:

        d = (c * t) / 2
    """

    speed = 340.0
    delay = 1.0

    expected_distance = 170.0

    distance = calculate_distance(
        delay,
        speed_of_sound=speed
    )

    assert distance == pytest.approx(
        expected_distance,
        rel=RTOL,
        abs=ATOL
    )


def test_calculate_distance_zero_delay():
    """
    Un retardo igual a cero debe corresponder a una distancia cero.
    """

    distance = calculate_distance(
        0.0,
        speed_of_sound=343.0
    )

    assert distance == pytest.approx(
        0.0,
        rel=RTOL,
        abs=ATOL
    )


def test_calculate_distance_fails_on_negative_delay():
    """
    Un retardo negativo violaría la causalidad del sistema.
    """

    with pytest.raises(ValueError):
        calculate_distance(
            -0.5,
            speed_of_sound=343.0
        )


@pytest.mark.parametrize(
    "invalid_delay",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calculate_distance_fails_on_non_finite_delay(
    invalid_delay
):
    """
    NaN e infinito no representan tiempos de vuelo válidos.
    """

    with pytest.raises(ValueError):
        calculate_distance(
            invalid_delay,
            speed_of_sound=343.0
        )


@pytest.mark.parametrize(
    "invalid_speed",
    [
        0.0,
        -10.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calculate_distance_fails_on_invalid_speed(
    invalid_speed
):
    """
    La velocidad debe ser finita y estrictamente positiva.
    """

    with pytest.raises(ValueError):
        calculate_distance(
            0.1,
            speed_of_sound=invalid_speed
        )


# ============================================================
# ESTIMATE RANGE
# ============================================================

def test_estimate_range_end_to_end():
    """
    Prueba de integración:

        muestras
        -> segundos
        -> velocidad según temperatura
        -> distancia
    """

    sample_rate = 48000.0
    delay_samples = 4800
    temperature = 20.0

    # 4800 / 48000 = 0.1 segundos
    #
    # c(20 °C) = 343.42 m/s
    #
    # d = (343.42 * 0.1) / 2
    #   = 17.171 m

    expected_distance = 17.171

    calculated_distance = estimate_range(
        delay_samples,
        sample_rate,
        temperature
    )

    assert calculated_distance == pytest.approx(
        expected_distance,
        rel=RTOL,
        abs=ATOL
    )


def test_estimate_range_zero_delay():
    """
    Cero muestras de retardo deben producir cero metros.
    """

    distance = estimate_range(
        delay_samples=0,
        sample_rate=48000.0,
        temperature_c=20.0
    )

    assert distance == pytest.approx(
        0.0,
        rel=RTOL,
        abs=ATOL
    )


def test_estimate_range_fails_on_negative_delay():
    """
    El nivel superior también debe rechazar retardos negativos.
    """

    with pytest.raises(ValueError):
        estimate_range(
            delay_samples=-100,
            sample_rate=48000.0,
            temperature_c=20.0
        )


@pytest.mark.parametrize(
    "invalid_delay",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_estimate_range_fails_on_non_finite_delay(
    invalid_delay
):
    """
    El retardo en muestras debe ser finito.
    """

    with pytest.raises(ValueError):
        estimate_range(
            delay_samples=invalid_delay,
            sample_rate=48000.0,
            temperature_c=20.0
        )


@pytest.mark.parametrize(
    "invalid_sample_rate",
    [
        0.0,
        -48000.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_estimate_range_fails_on_invalid_sample_rate(
    invalid_sample_rate
):
    """
    Fs debe ser finita y estrictamente positiva.
    """

    with pytest.raises(ValueError):
        estimate_range(
            delay_samples=1000,
            sample_rate=invalid_sample_rate,
            temperature_c=20.0
        )


# ============================================================
# CASO REAL DEL EXPERIMENTO
# ============================================================

def test_estimate_range_project_case_7500_samples():
    """
    Valida exactamente el escenario usado en el experimento
    de detección de ecos.

    Datos:

        Fs = 48000 Hz
        delay = 7500 muestras
        T = 20 °C

    Tiempo de vuelo:

        t = 7500 / 48000
          = 0.15625 s

    Velocidad:

        c = 343.42 m/s

    Distancia:

        d = (343.42 * 0.15625) / 2
          = 26.8296875 m
    """

    delay_samples = 7500
    sample_rate = 48000.0
    temperature = 20.0

    expected_distance = 26.8296875

    calculated_distance = estimate_range(
        delay_samples,
        sample_rate,
        temperature
    )

    assert calculated_distance == pytest.approx(
        expected_distance,
        rel=RTOL,
        abs=ATOL
    )