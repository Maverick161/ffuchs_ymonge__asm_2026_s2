
"""
Módulo: test_transforms.py
--------------------------
Responsabilidad: Validar matemáticamente nuestras implementaciones propias
de la DFT y FFT mediante señales de comportamiento espectral conocido,
comparaciones cruzadas entre ambos algoritmos y comparación contra
numpy.fft.fft como implementación de referencia.

También se verifican casos límite y entradas inválidas para asegurar
que los algoritmos tengan un comportamiento predecible y robusto.

Para ejecutar:
pytest tests/test_transforms.py
"""

import numpy as np  # Importamos NumPy para generar señales de prueba y utilizar su FFT como referencia
import pytest  # Importamos pytest para ejecutar pruebas automatizadas y comprobar excepciones

from radar.transforms.dft import compute_dft  # Importamos nuestra implementación propia de la DFT
from radar.transforms.fft import compute_fft  # Importamos nuestra implementación propia de la FFT


# Definimos explícitamente las tolerancias numéricas que utilizaremos en todas las comparaciones.
# Debido a que DFT, FFT y NumPy pueden realizar las operaciones en diferente orden,
# pueden existir pequeñas diferencias causadas por errores de redondeo de punto flotante.
RTOL = 1e-10
ATOL = 1e-12


def test_dft_impulse():
    """
    Prueba teórica:
    La DFT de un impulso discreto [1, 0, 0, 0] debe producir
    un espectro completamente plano [1, 1, 1, 1].

    Matemáticamente:
    δ[n] <-> 1
    """

    # Creamos un impulso discreto de longitud N=4
    x = np.array([1, 0, 0, 0], dtype=float)

    # Calculamos su DFT utilizando nuestra implementación
    X_dft = compute_dft(x)

    # El resultado teórico esperado es un valor 1 en todos los bins frecuenciales
    expected = np.array([1, 1, 1, 1], dtype=np.complex128)

    # Comparamos utilizando tolerancia numérica para evitar problemas por punto flotante
    assert np.allclose(X_dft, expected, rtol=RTOL, atol=ATOL)


def test_fft_impulse():
    """
    Prueba teórica:
    La FFT de un impulso discreto [1, 0, 0, 0] debe producir
    exactamente el mismo espectro teórico que la DFT.

    Matemáticamente:
    δ[n] <-> 1
    """

    # Creamos el mismo impulso discreto utilizado en la prueba anterior
    x = np.array([1, 0, 0, 0], dtype=float)

    # Calculamos su FFT utilizando nuestra implementación
    X_fft = compute_fft(x)

    # Definimos el resultado teórico esperado
    expected = np.array([1, 1, 1, 1], dtype=np.complex128)

    # Verificamos que nuestra FFT produzca el resultado esperado
    assert np.allclose(X_fft, expected, rtol=RTOL, atol=ATOL)


def test_dft_constant_signal():
    """
    Prueba teórica:
    Una señal constante [1, 1, 1, 1] contiene únicamente componente DC.

    Por lo tanto, para N=4:
    X[0] = 4
    X[1] = X[2] = X[3] = 0
    """

    # Creamos una señal constante de cuatro muestras
    x = np.array([1, 1, 1, 1], dtype=float)

    # Calculamos su DFT
    X_dft = compute_dft(x)

    # Toda la energía debe aparecer en el bin de frecuencia cero (DC)
    expected = np.array([4, 0, 0, 0], dtype=np.complex128)

    # Verificamos el resultado contra la solución teórica
    assert np.allclose(X_dft, expected, rtol=RTOL, atol=ATOL)


def test_fft_constant_signal():
    """
    Prueba teórica:
    Verifica mediante nuestra FFT que una señal constante
    únicamente contenga componente DC.
    """

    # Creamos una señal constante de longitud N=4
    x = np.array([1, 1, 1, 1], dtype=float)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # El resultado esperado contiene únicamente la componente DC
    expected = np.array([4, 0, 0, 0], dtype=np.complex128)

    # Verificamos el resultado
    assert np.allclose(X_fft, expected, rtol=RTOL, atol=ATOL)


def test_dft_zero_signal():
    """
    Prueba teórica:
    La transformada de una señal completamente nula debe ser
    también completamente nula en el dominio frecuencial.
    """

    # Creamos una señal sin ninguna componente
    x = np.array([0, 0, 0, 0], dtype=float)

    # Calculamos nuestra DFT
    X_dft = compute_dft(x)

    # Esperamos únicamente ceros complejos
    expected = np.zeros(4, dtype=np.complex128)

    # Comprobamos el resultado
    assert np.allclose(X_dft, expected, rtol=RTOL, atol=ATOL)


def test_fft_zero_signal():
    """
    Prueba teórica:
    La FFT de una señal completamente nula también debe producir
    únicamente ceros en todos los bins de frecuencia.
    """

    # Creamos una señal de ceros
    x = np.array([0, 0, 0, 0], dtype=float)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Definimos el resultado esperado
    expected = np.zeros(4, dtype=np.complex128)

    # Comprobamos el resultado
    assert np.allclose(X_fft, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64])
def test_dft_vs_numpy_random_signal(N):
    """
    Prueba de referencia:
    Compara directamente nuestra implementación de la DFT contra
    numpy.fft.fft para diferentes tamaños de entrada.

    Aunque NumPy ejecuta internamente algoritmos optimizados,
    matemáticamente debe producir la misma Transformada Discreta de Fourier.
    """

    # Creamos un generador pseudoaleatorio con semilla fija.
    # Esto garantiza que la prueba siempre utilice los mismos datos
    # y pueda reproducirse exactamente si alguna ejecución falla.
    rng = np.random.default_rng(42)

    # Generamos una señal real aleatoria de longitud N
    x = rng.random(N)

    # Calculamos nuestra implementación directa de la DFT
    X_custom_dft = compute_dft(x)

    # Calculamos la transformación utilizando NumPy como referencia
    X_numpy_fft = np.fft.fft(x)

    # Verificamos que ambas transformadas sean numéricamente equivalentes
    assert np.allclose(
        X_custom_dft,
        X_numpy_fft,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_fft_vs_numpy_random_signal(N):
    """
    Prueba de referencia principal:
    Compara nuestra FFT contra numpy.fft.fft utilizando diferentes
    tamaños N que son potencia de dos.

    Esta prueba ayuda a detectar errores que podrían aparecer únicamente
    en determinados niveles de la recursión del algoritmo Radix-2.
    """

    # Utilizamos nuevamente un generador con semilla fija
    # para garantizar que las pruebas sean reproducibles.
    rng = np.random.default_rng(12345)

    # Generamos N muestras aleatorias
    x = rng.random(N)

    # Ejecutamos nuestra implementación de la FFT
    X_custom_fft = compute_fft(x)

    # Ejecutamos la implementación de referencia de NumPy
    X_numpy_fft = np.fft.fft(x)

    # Verificamos equivalencia numérica dentro de las tolerancias definidas
    assert np.allclose(
        X_custom_fft,
        X_numpy_fft,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64])
def test_fft_vs_dft_custom(N):
    """
    Comparación cruzada:
    Verifica que nuestras implementaciones propias de DFT y FFT
    produzcan resultados numéricamente equivalentes.

    Matemáticamente ambos algoritmos calculan exactamente la misma
    transformada; la diferencia principal está en la eficiencia computacional.
    """

    # Creamos una señal reproducible mediante una semilla fija
    rng = np.random.default_rng(2026)

    # Generamos una señal real aleatoria de longitud N
    x = rng.random(N)

    # Calculamos la transformación utilizando nuestra DFT directa
    X_dft = compute_dft(x)

    # Calculamos la misma transformación utilizando nuestra FFT
    X_fft = compute_fft(x)

    # Verificamos que los resultados sean numéricamente equivalentes
    assert np.allclose(
        X_dft,
        X_fft,
        rtol=RTOL,
        atol=ATOL
    )


def test_transforms_complex_signal():
    """
    Prueba con números complejos:
    La definición matemática de la DFT admite señales complejas,
    por lo que nuestras implementaciones no deberían estar limitadas
    únicamente a entradas reales.

    Esta prueba compara DFT y FFT propias contra NumPy.
    """

    # Creamos una señal compleja de longitud N=4
    x = np.array([
        1 + 2j,
        3 - 1j,
        -2 + 0.5j,
        4 + 3j
    ], dtype=np.complex128)

    # Calculamos nuestra DFT
    X_dft = compute_dft(x)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Calculamos la transformada de referencia utilizando NumPy
    X_numpy = np.fft.fft(x)

    # Verificamos nuestra DFT contra NumPy
    assert np.allclose(
        X_dft,
        X_numpy,
        rtol=RTOL,
        atol=ATOL
    )

    # Verificamos nuestra FFT contra NumPy
    assert np.allclose(
        X_fft,
        X_numpy,
        rtol=RTOL,
        atol=ATOL
    )


def test_fft_known_sine_frequency():
    """
    Prueba relacionada directamente con procesamiento de señales:
    Generamos una senoide cuya frecuencia coincide exactamente con
    uno de los bins de la FFT.

    Para una señal real senoidal deben aparecer dos componentes
    espectrales principales: una en +f y otra en -f.
    """

    # Definimos la frecuencia de muestreo en Hertz
    sample_rate = 1024

    # Utilizamos N=1024, que es una potencia de dos compatible con nuestra FFT
    N = 1024

    # Definimos una frecuencia de 100 Hz que coincide exactamente
    # con el bin número 100 porque Delta_f = Fs/N = 1 Hz
    frequency = 100

    # Construimos el vector temporal discreto
    t = np.arange(N) / sample_rate

    # Generamos una senoide pura de 100 Hz
    x = np.sin(2 * np.pi * frequency * t)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Calculamos la magnitud del espectro
    magnitude = np.abs(X_fft)

    # Obtenemos los índices de los dos bins con mayor magnitud.
    # Para una senoide real esperamos componentes en +100 Hz y -100 Hz.
    dominant_bins = np.argsort(magnitude)[-2:]

    # Los bins esperados son 100 y N-100 = 924.
    # Ordenamos ambos arreglos para no depender del orden devuelto por argsort.
    expected_bins = np.array([frequency, N - frequency])

    # Verificamos que los dos picos principales estén en las frecuencias esperadas
    assert np.array_equal(
        np.sort(dominant_bins),
        np.sort(expected_bins)
    )


def test_fft_base_case_single_sample():
    """
    Prueba del caso base:
    Una FFT Radix-2 recursiva debe poder procesar correctamente
    una señal de una única muestra.

    Para N=1, la transformada es simplemente el mismo valor de entrada.
    """

    # Creamos una señal formada por una única muestra
    x = np.array([5.0])

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # El resultado teórico debe contener exactamente la misma muestra
    expected = np.array([5.0 + 0j])

    # Verificamos el caso base de la recursión
    assert np.allclose(
        X_fft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize("N", [3, 5, 6, 10, 1000])
def test_fft_fails_on_non_power_of_two(N):
    """
    Prueba de validación:
    Nuestra implementación Cooley-Tukey Radix-2 requiere que
    el tamaño N de la entrada sea una potencia de dos.

    Se prueban diferentes tamaños inválidos para verificar que
    la validación no funcione únicamente para un caso particular.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(42)

    # Generamos una señal cuyo tamaño deliberadamente NO es potencia de dos
    x = rng.random(N)

    # Esperamos que nuestra FFT rechace la entrada mediante ValueError.
    # Si la excepción ocurre, la prueba es exitosa.
    with pytest.raises(ValueError):
        compute_fft(x)


def test_dft_fails_on_empty_input():
    """
    Prueba de entrada inválida:
    Una DFT con N=0 no representa una transformación válida
    para nuestro sistema, por lo que debe rechazarse explícitamente.
    """

    # Creamos un arreglo vacío
    x = np.array([])

    # Esperamos que la implementación detecte la entrada inválida
    with pytest.raises(ValueError):
        compute_dft(x)


def test_fft_fails_on_empty_input():
    """
    Prueba de entrada inválida:
    Una FFT con N=0 debe rechazarse antes de intentar ejecutar
    el algoritmo recursivo.
    """

    # Creamos un arreglo vacío
    x = np.array([])

    # Esperamos que nuestra FFT genere una excepción controlada
    with pytest.raises(ValueError):
        compute_fft(x)


def test_dft_fails_on_two_dimensional_input():
    """
    Prueba de dimensionalidad:
    Nuestra implementación de DFT está diseñada para señales
    unidimensionales x[n], no para matrices bidimensionales.
    """

    # Creamos deliberadamente una matriz 2D
    x = np.array([
        [1, 2],
        [3, 4]
    ])

    # Esperamos que la implementación rechace este tipo de entrada
    with pytest.raises(ValueError):
        compute_dft(x)


def test_fft_fails_on_two_dimensional_input():
    """
    Prueba de dimensionalidad:
    Nuestra FFT está diseñada para procesar únicamente vectores
    unidimensionales de muestras.
    """

    # Creamos una matriz 2x2 que no representa una señal 1D válida
    x = np.array([
        [1, 2],
        [3, 4]
    ])

    # Esperamos que la implementación detecte y rechace la entrada
    with pytest.raises(ValueError):
        compute_fft(x)

