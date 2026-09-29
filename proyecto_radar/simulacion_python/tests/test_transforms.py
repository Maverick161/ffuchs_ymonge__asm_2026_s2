
"""
Módulo: test_transforms.py
--------------------------
Responsabilidad: Validar matemáticamente nuestras implementaciones propias
de la DFT, FFT e IFFT mediante señales de comportamiento espectral conocido,
comparaciones cruzadas entre los algoritmos y comparación contra las
implementaciones de referencia proporcionadas por NumPy.

También se verifican:
- Casos teóricos conocidos.
- Señales reales y complejas.
- Diferentes tamaños de entrada.
- Propiedades de transformada directa e inversa.
- Recuperación de señales mediante FFT/IFFT.
- Casos límite.
- Entradas inválidas.
- Inmutabilidad de los datos de entrada.

El objetivo es asegurar que los algoritmos tengan un comportamiento
matemáticamente correcto, predecible y robusto.

Para ejecutar:

pytest tests/test_transforms.py
"""

import numpy as np  # Importamos NumPy para generar señales de prueba y utilizar FFT/IFFT como referencia
import pytest  # Importamos pytest para ejecutar pruebas automatizadas y comprobar excepciones

from radar.transforms.dft import compute_dft  # Importamos nuestra implementación propia de la DFT
from radar.transforms.fft import compute_fft  # Importamos nuestra implementación propia de la FFT
from radar.transforms.ifft import compute_ifft  # Importamos nuestra implementación propia de la IFFT


# Definimos explícitamente las tolerancias numéricas que utilizaremos
# en todas las comparaciones.
#
# Debido a que DFT, FFT, IFFT y NumPy pueden realizar las operaciones
# matemáticas en diferente orden, pueden existir pequeñas diferencias
# causadas por errores de redondeo de punto flotante.
RTOL = 1e-10
ATOL = 1e-12


# =============================================================================
# PRUEBAS DE LA DFT
# =============================================================================


def test_dft_impulse():
    """
    Prueba teórica:

    La DFT de un impulso discreto:

        [1, 0, 0, 0]

    debe producir un espectro completamente plano:

        [1, 1, 1, 1]

    Matemáticamente:

        δ[n] <-> 1
    """

    # Creamos un impulso discreto de longitud N=4
    x = np.array([
        1,
        0,
        0,
        0
    ], dtype=float)

    # Calculamos su DFT utilizando nuestra implementación
    X_dft = compute_dft(x)

    # El resultado teórico esperado es un valor 1
    # en todos los bins frecuenciales
    expected = np.array([
        1,
        1,
        1,
        1
    ], dtype=np.complex128)

    # Comparamos utilizando tolerancia numérica
    assert np.allclose(
        X_dft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_dft_constant_signal():
    """
    Prueba teórica:

    Una señal constante:

        [1, 1, 1, 1]

    contiene únicamente componente DC.

    Por lo tanto, para N=4:

        X[0] = 4

        X[1] = X[2] = X[3] = 0
    """

    # Creamos una señal constante de cuatro muestras
    x = np.array([
        1,
        1,
        1,
        1
    ], dtype=float)

    # Calculamos su DFT
    X_dft = compute_dft(x)

    # Toda la energía debe aparecer en el bin de frecuencia cero
    expected = np.array([
        4,
        0,
        0,
        0
    ], dtype=np.complex128)

    # Verificamos el resultado contra la solución teórica
    assert np.allclose(
        X_dft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_dft_zero_signal():
    """
    Prueba teórica:

    La transformada de una señal completamente nula debe ser
    también completamente nula en el dominio frecuencial.
    """

    # Creamos una señal de ceros
    x = np.array([
        0,
        0,
        0,
        0
    ], dtype=float)

    # Calculamos nuestra DFT
    X_dft = compute_dft(x)

    # Esperamos únicamente ceros complejos
    expected = np.zeros(
        4,
        dtype=np.complex128
    )

    # Comprobamos el resultado
    assert np.allclose(
        X_dft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64
    ]
)
def test_dft_vs_numpy_random_signal(N):
    """
    Prueba de referencia:

    Compara directamente nuestra implementación de la DFT contra:

        numpy.fft.fft()

    para diferentes tamaños de entrada.

    Aunque NumPy ejecuta internamente algoritmos optimizados,
    matemáticamente debe producir la misma Transformada Discreta
    de Fourier.
    """

    # Creamos un generador pseudoaleatorio con semilla fija
    # para garantizar reproducibilidad.
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


# =============================================================================
# PRUEBAS DE LA FFT
# =============================================================================


def test_fft_impulse():
    """
    Prueba teórica:

    La FFT de un impulso discreto:

        [1, 0, 0, 0]

    debe producir exactamente:

        [1, 1, 1, 1]

    Matemáticamente:

        δ[n] <-> 1
    """

    # Creamos un impulso discreto
    x = np.array([
        1,
        0,
        0,
        0
    ], dtype=float)

    # Calculamos su FFT
    X_fft = compute_fft(x)

    # Definimos el resultado teórico esperado
    expected = np.array([
        1,
        1,
        1,
        1
    ], dtype=np.complex128)

    # Verificamos el resultado
    assert np.allclose(
        X_fft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_fft_constant_signal():
    """
    Prueba teórica:

    Verifica mediante nuestra FFT que una señal constante
    únicamente contenga componente DC.
    """

    # Creamos una señal constante de longitud N=4
    x = np.array([
        1,
        1,
        1,
        1
    ], dtype=float)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # El resultado esperado contiene únicamente DC
    expected = np.array([
        4,
        0,
        0,
        0
    ], dtype=np.complex128)

    # Verificamos el resultado
    assert np.allclose(
        X_fft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_fft_zero_signal():
    """
    Prueba teórica:

    La FFT de una señal completamente nula también debe producir
    únicamente ceros en todos los bins de frecuencia.
    """

    # Creamos una señal de ceros
    x = np.array([
        0,
        0,
        0,
        0
    ], dtype=float)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Definimos el resultado esperado
    expected = np.zeros(
        4,
        dtype=np.complex128
    )

    # Comprobamos el resultado
    assert np.allclose(
        X_fft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256
    ]
)
def test_fft_vs_numpy_random_signal(N):
    """
    Prueba de referencia principal:

    Compara nuestra FFT contra:

        numpy.fft.fft()

    utilizando diferentes tamaños N que son potencia de dos.

    Esta prueba ayuda a detectar errores que podrían aparecer únicamente
    en determinados niveles de la recursión Radix-2.
    """

    # Utilizamos un generador con semilla fija
    rng = np.random.default_rng(12345)

    # Generamos N muestras aleatorias
    x = rng.random(N)

    # Ejecutamos nuestra implementación FFT
    X_custom_fft = compute_fft(x)

    # Ejecutamos la implementación de referencia de NumPy
    X_numpy_fft = np.fft.fft(x)

    # Verificamos equivalencia numérica
    assert np.allclose(
        X_custom_fft,
        X_numpy_fft,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64
    ]
)
def test_fft_vs_dft_custom(N):
    """
    Comparación cruzada:

    Verifica que nuestras implementaciones propias de DFT y FFT
    produzcan resultados numéricamente equivalentes.

    Matemáticamente ambos algoritmos calculan la misma transformada.

    La diferencia principal está en la eficiencia computacional.
    """

    # Creamos una señal reproducible
    rng = np.random.default_rng(2026)

    # Generamos una señal real aleatoria
    x = rng.random(N)

    # Calculamos nuestra DFT
    X_dft = compute_dft(x)

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Comparamos ambos resultados
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
    por lo que nuestras implementaciones no deben estar limitadas
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

    # Calculamos la referencia NumPy
    X_numpy = np.fft.fft(x)

    # Verificamos DFT contra NumPy
    assert np.allclose(
        X_dft,
        X_numpy,
        rtol=RTOL,
        atol=ATOL
    )

    # Verificamos FFT contra NumPy
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
    espectrales principales:

        +f
        -f
    """

    # Definimos la frecuencia de muestreo
    sample_rate = 1024

    # Utilizamos N=1024, potencia de dos
    N = 1024

    # Elegimos una frecuencia que coincide exactamente
    # con el bin número 100.
    frequency = 100

    # Construimos el vector temporal
    t = np.arange(N) / sample_rate

    # Generamos una senoide pura
    x = np.sin(
        2 * np.pi * frequency * t
    )

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # Calculamos la magnitud
    magnitude = np.abs(X_fft)

    # Obtenemos los dos bins de mayor magnitud
    dominant_bins = np.argsort(
        magnitude
    )[-2:]

    # Para una senoide real esperamos:
    #
    # +100 Hz -> bin 100
    # -100 Hz -> bin N - 100 = 924
    expected_bins = np.array([
        frequency,
        N - frequency
    ])

    # Verificamos los bins principales
    assert np.array_equal(
        np.sort(dominant_bins),
        np.sort(expected_bins)
    )


def test_fft_base_case_single_sample():
    """
    Prueba del caso base:

    Una FFT Radix-2 recursiva debe poder procesar correctamente
    una señal formada por una única muestra.

    Para N=1:

        FFT([x0]) = [x0]
    """

    # Creamos una señal de una única muestra
    x = np.array([
        5.0
    ])

    # Calculamos nuestra FFT
    X_fft = compute_fft(x)

    # El resultado debe ser idéntico
    expected = np.array([
        5.0 + 0j
    ])

    # Verificamos el caso base
    assert np.allclose(
        X_fft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


# =============================================================================
# PRUEBAS DE LA IFFT
# =============================================================================


def test_ifft_dc_spectrum_produces_constant_signal():
    """
    Prueba teórica fundamental de la IFFT.

    Un espectro que contiene únicamente componente DC:

        X = [4, 0, 0, 0]

    debe producir mediante IFFT:

        x = [1, 1, 1, 1]

    Matemáticamente:

        x[n] = (1/N) * X[0]

    Esta prueba permite verificar directamente:

    - La definición de la IFFT.
    - El factor de normalización 1/N.
    - El manejo correcto de DC.
    """

    # Creamos un espectro con únicamente componente DC
    X = np.array([
        4.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j
    ], dtype=np.complex128)

    # Calculamos nuestra IFFT
    x_ifft = compute_ifft(X)

    # El resultado esperado es una señal constante
    expected = np.array([
        1.0,
        1.0,
        1.0,
        1.0
    ], dtype=np.complex128)

    # Comparamos directamente el resultado complejo completo
    assert np.allclose(
        x_ifft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_ifft_flat_spectrum_produces_impulse():
    """
    Prueba teórica complementaria.

    Un espectro completamente plano:

        X = [1, 1, 1, 1]

    debe producir un impulso temporal:

        x = [1, 0, 0, 0]

    Esta prueba permite detectar errores en:

    - La normalización 1/N.
    - El signo de la transformada inversa.
    - La estructura general del algoritmo.
    """

    # Creamos un espectro plano
    X = np.array([
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j
    ], dtype=np.complex128)

    # Calculamos nuestra IFFT
    x_ifft = compute_ifft(X)

    # Resultado esperado
    expected = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ], dtype=np.complex128)

    # Verificamos
    assert np.allclose(
        x_ifft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


def test_ifft_zero_spectrum():
    """
    Verifica el comportamiento frente a un espectro completamente nulo.

    Matemáticamente:

        X[k] = 0

    para todos los k implica:

        x[n] = 0

    para todos los n.
    """

    # Creamos un espectro nulo de longitud N=8
    X = np.zeros(
        8,
        dtype=np.complex128
    )

    # Calculamos la IFFT
    x_ifft = compute_ifft(X)

    # Resultado esperado
    expected = np.zeros(
        8,
        dtype=np.complex128
    )

    # Comparamos
    assert np.allclose(
        x_ifft,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256
    ]
)
def test_ifft_vs_numpy_random_complex_spectrum(N):
    """
    PRUEBA DE REFERENCIA PRINCIPAL PARA LA IFFT.

    Compara nuestra implementación directamente contra:

        numpy.fft.ifft()

    utilizando espectros complejos generados independientemente.

    Esta prueba es especialmente importante porque NO utiliza
    nuestra propia compute_fft() para crear el resultado esperado.

    De esta manera evitamos que posibles errores compatibles entre
    FFT e IFFT se oculten entre sí.

    También se prueban diferentes tamaños potencia de dos para
    cubrir distintos niveles de recursión Radix-2.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        2026 + N
    )

    # Generamos independientemente la parte real
    real_part = rng.random(N)

    # Generamos independientemente la parte imaginaria
    imaginary_part = rng.random(N)

    # Construimos un espectro complejo arbitrario
    X = (
        real_part
        + 1j * imaginary_part
    )

    # Calculamos nuestra IFFT
    x_custom = compute_ifft(X)

    # Calculamos NumPy como referencia independiente
    x_numpy = np.fft.ifft(X)

    # Verificamos equivalencia numérica
    assert np.allclose(
        x_custom,
        x_numpy,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128
    ]
)
def test_ifft_recovers_original_real_signal(N):
    """
    Verifica la identidad:

        IFFT(FFT(x)) = x

    utilizando señales temporales reales.

    No se elimina la parte imaginaria mediante np.real().

    Esto obliga a que cualquier componente imaginaria residual
    producida por error también se mantenga dentro de las
    tolerancias numéricas.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        5000 + N
    )

    # Generamos una señal temporal real
    original_time_domain = rng.random(N)

    # Transformamos al dominio frecuencial
    complex_spectrum = compute_fft(
        original_time_domain
    )

    # Regresamos al dominio temporal
    recovered_time_domain = compute_ifft(
        complex_spectrum
    )

    # Comparamos directamente el resultado completo
    assert np.allclose(
        recovered_time_domain,
        original_time_domain,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64
    ]
)
def test_ifft_recovers_original_complex_signal(N):
    """
    Verifica la identidad:

        IFFT(FFT(x)) = x

    para señales temporales COMPLEJAS.

    La IFFT no está limitada matemáticamente a señales reales.

    Este comportamiento también es necesario para la correlación
    mediante FFT utilizada posteriormente en el radar.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        7000 + N
    )

    # Generamos parte real
    real_part = rng.random(N)

    # Generamos parte imaginaria
    imaginary_part = rng.random(N)

    # Construimos la señal compleja
    original_signal = (
        real_part
        + 1j * imaginary_part
    )

    # Transformamos utilizando nuestra FFT
    spectrum = compute_fft(
        original_signal
    )

    # Aplicamos nuestra IFFT
    recovered_signal = compute_ifft(
        spectrum
    )

    # Debemos recuperar tanto parte real como imaginaria
    assert np.allclose(
        recovered_signal,
        original_signal,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128
    ]
)
def test_fft_recovers_original_spectrum_after_ifft(N):
    """
    Verifica la identidad complementaria:

        FFT(IFFT(X)) = X

    Esta prueba comienza directamente desde un espectro
    complejo arbitrario.

    Así verificamos ambas direcciones:

        x -> FFT -> IFFT -> x

    y:

        X -> IFFT -> FFT -> X
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        9000 + N
    )

    # Construimos un espectro complejo
    original_spectrum = (
        rng.random(N)
        + 1j * rng.random(N)
    )

    # Convertimos al dominio temporal
    time_domain = compute_ifft(
        original_spectrum
    )

    # Regresamos al dominio frecuencial
    recovered_spectrum = compute_fft(
        time_domain
    )

    # Verificamos que recuperemos el espectro original
    assert np.allclose(
        recovered_spectrum,
        original_spectrum,
        rtol=RTOL,
        atol=ATOL
    )


def test_ifft_known_complex_input_vs_numpy():
    """
    Verifica un espectro complejo explícito contra NumPy.

    Esta prueba complementa los casos pseudoaleatorios con
    un ejemplo fijo y fácilmente reproducible.
    """

    # Construimos un espectro complejo conocido
    X = np.array([
        1.0 + 2.0j,
        3.0 - 1.0j,
        -2.0 + 0.5j,
        4.0 + 3.0j
    ], dtype=np.complex128)

    # Calculamos nuestra IFFT
    custom_ifft = compute_ifft(X)

    # Calculamos la referencia NumPy
    numpy_ifft = np.fft.ifft(X)

    # Comparamos
    assert np.allclose(
        custom_ifft,
        numpy_ifft,
        rtol=RTOL,
        atol=ATOL
    )


def test_ifft_base_case_single_sample():
    """
    Verifica explícitamente el caso base de la IFFT.

    Para:

        N = 1

    debe cumplirse:

        IFFT([X0]) = [X0]

    porque:

        1/N = 1
    """

    # Creamos una muestra compleja
    X = np.array([
        5.0 + 2.0j
    ])

    # Calculamos nuestra IFFT
    result = compute_ifft(X)

    # Resultado esperado
    expected = np.array([
        5.0 + 2.0j
    ])

    # Verificamos
    assert np.allclose(
        result,
        expected,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "N",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128
    ]
)
def test_ifft_preserves_output_length(N):
    """
    Verifica que la IFFT conserve exactamente el número
    de muestras de entrada.

    Si el espectro contiene N bins:

        len(IFFT(X)) = N
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        10000 + N
    )

    # Generamos un espectro complejo de longitud N
    X = (
        rng.random(N)
        + 1j * rng.random(N)
    )

    # Ejecutamos la IFFT
    result = compute_ifft(X)

    # La longitud debe conservarse
    assert len(result) == N


def test_ifft_does_not_modify_input():
    """
    Verifica una propiedad arquitectónica importante:

    compute_ifft() debe ser una operación NO destructiva.

    El espectro recibido como entrada debe permanecer
    exactamente igual después de la llamada.
    """

    # Creamos un espectro complejo conocido
    X = np.array([
        1.0 + 1.0j,
        2.0 - 3.0j,
        4.0 + 0.5j,
        -2.0 + 2.0j
    ], dtype=np.complex128)

    # Guardamos una copia independiente
    original_input = X.copy()

    # Ejecutamos nuestra IFFT
    compute_ifft(X)

    # El arreglo original no debe haberse modificado
    assert np.array_equal(
        X,
        original_input
    )


# =============================================================================
# PRUEBAS DE ENTRADAS INVÁLIDAS — DFT / FFT / IFFT
# =============================================================================


def test_dft_fails_on_empty_input():
    """
    Una DFT con N=0 no representa una transformación válida
    para nuestro sistema.
    """

    # Creamos un arreglo vacío
    x = np.array([])

    # Esperamos una excepción controlada
    with pytest.raises(ValueError):
        compute_dft(x)


def test_fft_fails_on_empty_input():
    """
    Una FFT con N=0 debe rechazarse antes de ejecutar
    el algoritmo recursivo.
    """

    # Creamos un arreglo vacío
    x = np.array([])

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_fft(x)


def test_ifft_fails_on_empty_input():
    """
    Una IFFT con N=0 tampoco representa una operación válida.

    La función debe detectar esta condición antes de intentar
    utilizar internamente la FFT.
    """

    # Creamos un espectro vacío
    X = np.array([])

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_ifft(X)


@pytest.mark.parametrize(
    "N",
    [
        3,
        5,
        6,
        10,
        1000
    ]
)
def test_fft_fails_on_non_power_of_two(N):
    """
    Nuestra FFT Cooley-Tukey Radix-2 requiere:

        N = 2^m

    Los tamaños indicados deliberadamente no cumplen
    esta condición.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(42)

    # Generamos una señal de tamaño inválido
    x = rng.random(N)

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_fft(x)


@pytest.mark.parametrize(
    "N",
    [
        3,
        5,
        6,
        10,
        1000
    ]
)
def test_ifft_fails_on_non_power_of_two(N):
    """
    La IFFT reutiliza la misma estructura Radix-2 que la FFT.

    Por lo tanto, también requiere:

        N = 2^m

    Los tamaños que no son potencia de dos deben rechazarse.
    """

    # Creamos un generador reproducible
    rng = np.random.default_rng(
        123 + N
    )

    # Generamos un espectro complejo de tamaño inválido
    X = (
        rng.random(N)
        + 1j * rng.random(N)
    )

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_ifft(X)


def test_dft_fails_on_two_dimensional_input():
    """
    Nuestra DFT está diseñada para señales unidimensionales x[n],
    no para matrices bidimensionales.
    """

    # Creamos deliberadamente una matriz 2D
    x = np.array([
        [1, 2],
        [3, 4]
    ])

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_dft(x)


def test_fft_fails_on_two_dimensional_input():
    """
    Nuestra FFT está diseñada para procesar únicamente
    vectores unidimensionales.
    """

    # Creamos una matriz 2x2
    x = np.array([
        [1, 2],
        [3, 4]
    ])

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_fft(x)


def test_ifft_fails_on_two_dimensional_input():
    """
    Nuestra IFFT está diseñada para procesar espectros
    unidimensionales X[k].

    No soporta matrices o transformadas multidimensionales.
    """

    # Creamos deliberadamente una matriz compleja 2D
    X = np.array([
        [
            1.0 + 0.0j,
            2.0 + 0.0j
        ],
        [
            3.0 + 0.0j,
            4.0 + 0.0j
        ]
    ])

    # Esperamos ValueError
    with pytest.raises(ValueError):
        compute_ifft(X)

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

def test_ifft_recovers_original_signal():
    """
    Verifica matemáticamente que IFFT(FFT(x)) == x.
    La transformada inversa debe recuperar la señal temporal original
    con precisión de punto flotante.
    """
    rng = np.random.default_rng(2026)
    
    # Usamos N=64 (potencia de 2) para nuestra FFT Radix-2
    original_time_domain = rng.random(64)
    
    # 1. Transformamos al dominio de la frecuencia
    complex_spectrum = compute_fft(original_time_domain)
    
    # 2. Devolvemos al dominio del tiempo usando nuestra IFFT
    recovered_time_domain = compute_ifft(complex_spectrum)
    
    # Verificamos ignorando la parte imaginaria residual (que debería ser ~0)
    recovered_real = np.real(recovered_time_domain)
    
    assert np.allclose(
        original_time_domain, 
        recovered_real, 
        rtol=RTOL, 
        atol=ATOL
    )