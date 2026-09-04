
"""
Módulo: test_signals.py
-----------------------
Responsabilidad: Verificar que la creación de señales sintéticas, la inyección
de ruido y las funciones auxiliares relacionadas con potencias de dos
funcionen según el comportamiento matemático esperado.

Las pruebas incluyen casos normales, casos límite, validación estadística
del ruido y comprobaciones para evitar modificaciones accidentales de las
señales originales.

Para ejecutar:
pytest tests/test_signals.py
"""

import numpy as np  # Importamos NumPy para operaciones numéricas, comparaciones y generación pseudoaleatoria
import pytest  # Importamos pytest para parametrizar pruebas y verificar excepciones

from radar.signals.generators import generate_sine, add_noise  # Importamos nuestros generadores de señales
from radar.utils.validation import is_power_of_two, next_power_of_two, pad_to_power_of_two  # Importamos las utilidades relacionadas con tamaños FFT


def test_generate_sine_properties():
    """
    Verifica las propiedades estructurales básicas de una señal senoidal:
    número de muestras, frecuencia de muestreo, duración y vector temporal.
    """

    # Definimos una frecuencia de muestreo de 1000 muestras por segundo
    fs = 1000.0

    # Definimos una duración total de 2 segundos
    duration = 2.0

    # Definimos una frecuencia senoidal de 10 Hz
    freq = 10.0

    # Generamos la señal utilizando nuestro generador
    signal = generate_sine(
        frequency=freq,
        sample_rate=fs,
        duration=duration
    )

    # Como N = Fs * T, esperamos exactamente 1000 * 2 = 2000 muestras
    assert signal.number_of_samples == 2000

    # Verificamos que la frecuencia de muestreo se conserve correctamente
    assert signal.sample_rate == fs

    # Verificamos la duración utilizando tolerancia de punto flotante
    assert np.isclose(signal.duration, duration)

    # El primer instante temporal debe corresponder a t = 0 segundos
    assert np.isclose(signal.time_vector[0], 0.0)

    # Como Fs = 1000 Hz, el período de muestreo es Ts = 1/Fs = 0.001 segundos
    assert np.isclose(signal.time_vector[1], 0.001)

    # El vector temporal debe contener exactamente una posición por cada muestra
    assert len(signal.time_vector) == signal.number_of_samples


def test_generate_sine_matches_equation():
    """
    Verifica que las muestras generadas realmente correspondan a la ecuación:

        x(t) = A * sin(2*pi*f*t + phi)

    Esta prueba es importante porque comprobar solamente tamaño y duración
    no garantiza que la señal generada sea matemáticamente una senoide.
    """

    # Definimos parámetros conocidos para la señal
    frequency = 25.0
    sample_rate = 1000.0
    duration = 0.1
    amplitude = 2.0
    phase = np.pi / 4

    # Generamos la señal utilizando nuestro módulo
    signal = generate_sine(
        frequency=frequency,
        sample_rate=sample_rate,
        duration=duration,
        amplitude=amplitude,
        phase=phase
    )

    # Construimos manualmente el vector temporal esperado
    t = np.arange(signal.number_of_samples) / sample_rate

    # Evaluamos directamente la ecuación matemática de la senoide
    expected_samples = amplitude * np.sin(
        2 * np.pi * frequency * t + phase
    )

    # Verificamos que todas las muestras generadas coincidan con la ecuación
    assert np.allclose(signal.samples, expected_samples)


def test_generate_sine_amplitude_and_phase():
    """
    Verifica específicamente el manejo de amplitud y fase.

    Si phi = pi/2, entonces para t = 0:

        x(0) = A * sin(pi/2) = A

    Por lo tanto, la primera muestra debe ser igual a la amplitud.
    """

    # Definimos una amplitud fácil de comprobar
    amplitude = 3.0

    # Definimos un desfase de pi/2 radianes
    phase = np.pi / 2

    # Generamos una senoide con dichos parámetros
    signal = generate_sine(
        frequency=10.0,
        sample_rate=1000.0,
        duration=1.0,
        amplitude=amplitude,
        phase=phase
    )

    # Para t=0 y phi=pi/2 esperamos x(0)=A
    assert np.isclose(signal.samples[0], amplitude)


@pytest.mark.parametrize(
    "sample_rate, duration",
    [
        (0.0, 1.0),
        (-1000.0, 1.0),
        (1000.0, 0.0),
        (1000.0, -1.0),
    ]
)
def test_generate_sine_fails_on_invalid_sampling_parameters(sample_rate, duration):
    """
    Verifica que el generador rechace frecuencias de muestreo y duraciones
    físicamente inválidas.

    Deben cumplirse:

        Fs > 0
        T > 0
    """

    # Esperamos que cualquier combinación inválida genere ValueError
    with pytest.raises(ValueError):
        generate_sine(
            frequency=10.0,
            sample_rate=sample_rate,
            duration=duration
        )


def test_generate_sine_allows_frequency_above_nyquist():
    """
    Verifica deliberadamente que el generador permita crear frecuencias
    superiores a Nyquist.

    Esto NO debe considerarse un error del generador porque necesitaremos
    este comportamiento para realizar experimentos de aliasing.

    Para Fs = 1000 Hz:
        frecuencia de Nyquist = 500 Hz

    Generamos deliberadamente una señal de 700 Hz.
    """

    # Definimos una frecuencia de muestreo de 1000 Hz
    fs = 1000.0

    # Definimos una frecuencia superior a Nyquist
    frequency = 700.0

    # El generador debe permitir crear la señal sin lanzar una excepción
    signal = generate_sine(
        frequency=frequency,
        sample_rate=fs,
        duration=1.0
    )

    # Verificamos que la señal se haya creado correctamente
    assert signal.number_of_samples == 1000


def test_add_noise_preserves_signal_properties():
    """
    Verifica que añadir ruido modifique las muestras pero conserve las
    propiedades estructurales fundamentales de la señal original.
    """

    # Generamos una señal senoidal de referencia
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Creamos un generador pseudoaleatorio con semilla fija
    # para garantizar que el experimento sea reproducible
    rng = np.random.default_rng(42)

    # Añadimos ruido blanco gaussiano con varianza conocida
    noisy_signal = add_noise(
        signal,
        noise_variance=0.5,
        rng=rng
    )

    # El ruido no debe modificar el número de muestras
    assert noisy_signal.number_of_samples == signal.number_of_samples

    # La frecuencia de muestreo debe permanecer exactamente igual
    assert noisy_signal.sample_rate == signal.sample_rate

    # Las muestras de la señal ruidosa deben ser diferentes de las originales
    assert not np.allclose(noisy_signal.samples, signal.samples)


def test_add_noise_does_not_modify_original_signal():
    """
    Verifica una propiedad arquitectónica importante:
    add_noise() debe generar una nueva señal y NO modificar la señal original.

    Esto permite comparar posteriormente la señal limpia contra la señal
    contaminada sin perder los datos originales.
    """

    # Generamos una señal limpia
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Guardamos una copia independiente de las muestras originales
    original_samples = signal.samples.copy()

    # Creamos un generador pseudoaleatorio reproducible
    rng = np.random.default_rng(123)

    # Generamos una nueva señal con ruido
    add_noise(
        signal,
        noise_variance=0.5,
        rng=rng
    )

    # Verificamos que las muestras de la señal original permanezcan intactas
    assert np.array_equal(signal.samples, original_samples)


def test_add_noise_is_reproducible_with_same_seed():
    """
    Verifica que utilizar la misma semilla pseudoaleatoria produzca
    exactamente la misma realización de ruido.

    Esto es importante para poder reproducir experimentos y resultados
    incluidos posteriormente en el informe.
    """

    # Generamos la misma señal base para ambas pruebas
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Creamos dos generadores independientes utilizando exactamente la misma semilla
    rng1 = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)

    # Añadimos ruido utilizando el primer generador
    noisy_signal_1 = add_noise(
        signal,
        noise_variance=0.5,
        rng=rng1
    )

    # Añadimos ruido nuevamente utilizando el segundo generador
    noisy_signal_2 = add_noise(
        signal,
        noise_variance=0.5,
        rng=rng2
    )

    # Al compartir la misma semilla, ambos resultados deben ser idénticos
    assert np.array_equal(
        noisy_signal_1.samples,
        noisy_signal_2.samples
    )


def test_add_noise_zero_variance():
    """
    Caso límite:
    Si la varianza del ruido es cero, entonces su desviación estándar
    también es cero y todas las muestras de ruido deben ser exactamente cero.

    Por lo tanto:

        señal_ruidosa = señal_original
    """

    # Generamos una señal limpia
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Creamos un generador pseudoaleatorio
    rng = np.random.default_rng(42)

    # Solicitamos ruido con varianza exactamente cero
    noisy_signal = add_noise(
        signal,
        noise_variance=0.0,
        rng=rng
    )

    # Las muestras deben permanecer numéricamente iguales
    assert np.allclose(
        noisy_signal.samples,
        signal.samples
    )


def test_add_noise_statistical_properties():
    """
    Verifica estadísticamente que el ruido añadido tenga aproximadamente
    la media y varianza solicitadas.

    Para ruido blanco gaussiano:

        media ≈ 0
        varianza ≈ noise_variance

    Como trabajamos con una realización aleatoria finita, no esperamos
    igualdad exacta sino una aproximación razonable.
    """

    # Utilizamos una señal suficientemente larga para obtener
    # estadísticas representativas del ruido
    signal = generate_sine(
        frequency=10,
        sample_rate=10000,
        duration=2.0
    )

    # Definimos la varianza que esperamos obtener
    expected_variance = 0.5

    # Utilizamos una semilla fija para que la prueba sea reproducible
    rng = np.random.default_rng(12345)

    # Añadimos ruido a la señal
    noisy_signal = add_noise(
        signal,
        noise_variance=expected_variance,
        rng=rng
    )

    # Recuperamos matemáticamente el ruido añadido:
    # ruido = señal contaminada - señal original
    noise = noisy_signal.samples - signal.samples

    # Calculamos la media experimental del ruido
    measured_mean = np.mean(noise)

    # Calculamos la varianza experimental del ruido
    measured_variance = np.var(noise)

    # La media debe encontrarse aproximadamente alrededor de cero
    assert np.isclose(
        measured_mean,
        0.0,
        atol=0.03
    )

    # La varianza debe encontrarse razonablemente cerca de la solicitada
    assert np.isclose(
        measured_variance,
        expected_variance,
        rtol=0.08
    )


def test_add_noise_fails_on_negative_variance():
    """
    Verifica que una varianza negativa sea rechazada.

    Matemáticamente la varianza debe cumplir:

        sigma² >= 0
    """

    # Generamos una señal válida
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Una varianza negativa no tiene significado válido para este generador
    with pytest.raises(ValueError):
        add_noise(
            signal,
            noise_variance=-0.1
        )


@pytest.mark.parametrize(
    "value",
    [1, 2, 4, 8, 16, 32, 64, 512, 1024]
)
def test_is_power_of_two_true(value):
    """
    Verifica diferentes valores que matemáticamente sí son potencias de dos.

    Recordemos que:

        1 = 2^0

    Por lo tanto, 1 también debe considerarse potencia de dos.
    """

    # Todos los valores parametrizados deben ser reconocidos correctamente
    assert is_power_of_two(value) is True


@pytest.mark.parametrize(
    "value",
    [0, -1, -2, 3, 5, 6, 7, 10, 1000]
)
def test_is_power_of_two_false(value):
    """
    Verifica valores que NO deben reconocerse como potencias válidas de dos.

    Incluimos cero y números negativos porque no representan tamaños
    válidos de una FFT para nuestro sistema.
    """

    # Todos estos valores deben producir False
    assert is_power_of_two(value) is False


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (15, 16),
        (16, 16),
        (17, 32),
        (511, 512),
        (512, 512),
        (513, 1024),
        (1000, 1024),
        (1024, 1024),
        (1025, 2048),
    ]
)
def test_next_power_of_two(value, expected):
    """
    Verifica el cálculo de la siguiente potencia de dos para valores
    ubicados antes, exactamente sobre y después de diferentes fronteras.

    Por ejemplo:

        15 -> 16
        16 -> 16
        17 -> 32
    """

    # Comparamos el resultado de nuestra función contra el valor esperado
    assert next_power_of_two(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, -1, -10]
)
def test_next_power_of_two_fails_on_non_positive_values(value):
    """
    Verifica que next_power_of_two() rechace tamaños menores o iguales a cero.

    Para nuestro sistema estos valores no representan una cantidad
    válida de muestras.
    """

    # Esperamos una excepción controlada para cada entrada inválida
    with pytest.raises(ValueError):
        next_power_of_two(value)


def test_zero_padding():
    """
    Verifica que el zero padding extienda correctamente una señal hasta
    la siguiente potencia de dos.

    Para N = 1000:

        siguiente potencia de dos = 1024

    Por lo tanto deben agregarse exactamente 24 ceros.
    """

    # Generamos una señal con exactamente 1000 muestras
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Aplicamos zero padding hasta la siguiente potencia de dos
    padded = pad_to_power_of_two(signal)

    # Verificamos que el nuevo tamaño sea N = 1024
    assert padded.number_of_samples == 1024

    # Verificamos que las primeras 1000 muestras sean exactamente
    # las mismas muestras que existían en la señal original
    assert np.array_equal(
        padded.samples[:1000],
        signal.samples
    )

    # Verificamos que las últimas 24 posiciones agregadas sean cero
    assert np.all(
        padded.samples[1000:] == 0.0
    )

    # El zero padding no debe modificar la frecuencia de muestreo
    assert padded.sample_rate == signal.sample_rate


def test_zero_padding_does_not_modify_original_signal():
    """
    Verifica que pad_to_power_of_two() produzca una nueva representación
    sin modificar las muestras originales.

    La señal física adquirida debe permanecer disponible sin alteraciones.
    """

    # Generamos una señal original de N=1000
    signal = generate_sine(
        frequency=10,
        sample_rate=1000,
        duration=1.0
    )

    # Guardamos una copia independiente de sus muestras
    original_samples = signal.samples.copy()

    # Ejecutamos el padding
    pad_to_power_of_two(signal)

    # La señal original debe seguir teniendo exactamente 1000 muestras
    assert signal.number_of_samples == 1000

    # Las muestras originales deben permanecer idénticas
    assert np.array_equal(
        signal.samples,
        original_samples
    )


def test_zero_padding_when_already_power_of_two():
    """
    Verifica el caso en el que la señal ya contiene un número de muestras
    que es potencia de dos.

    Para N=1024 no debe agregarse ningún cero adicional.
    """

    # Creamos una señal con exactamente 1024 muestras
    signal = generate_sine(
        frequency=10,
        sample_rate=1024,
        duration=1.0
    )

    # Aplicamos la función de zero padding
    padded = pad_to_power_of_two(signal)

    # El tamaño debe permanecer exactamente igual
    assert padded.number_of_samples == 1024

    # Las muestras deben conservarse sin modificaciones
    assert np.array_equal(
        padded.samples,
        signal.samples
    )

    # La frecuencia de muestreo también debe conservarse
    assert padded.sample_rate == signal.sample_rate

