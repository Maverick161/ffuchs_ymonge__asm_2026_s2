
"""
Módulo: test_spectrum.py
------------------------
Responsabilidad: Validar que el análisis del espectro complejo permita
obtener correctamente el eje de frecuencias, la magnitud normalizada,
la fase y el espectro unilateral.

Las pruebas combinan:
- Casos matemáticos conocidos.
- Comparaciones contra funciones de referencia de NumPy.
- Señales sintéticas con comportamiento espectral conocido.
- Casos límite y entradas inválidas.
- Comprobaciones de consistencia interna del objeto Spectrum.

Para ejecutar:
pytest tests/test_spectrum.py
"""

import numpy as np  # Importamos NumPy para cálculos numéricos y funciones espectrales de referencia
import pytest  # Importamos pytest para pruebas parametrizadas y comprobación de excepciones

from radar.signals.signal import Signal  # Importamos la estructura base para representar señales
from radar.signals.generators import generate_sine  # Importamos el generador de señales senoidales
from radar.transforms.fft import compute_fft  # Importamos nuestra implementación propia de FFT
from radar.analysis.spectrum import compute_spectrum  # Importamos el análisis del espectro complejo


# Definimos explícitamente las tolerancias utilizadas en las comparaciones
# numéricas para contemplar pequeños errores de punto flotante.
RTOL = 1e-10
ATOL = 1e-12


def test_spectrum_frequency_axis():
    """
    Verifica que el eje de frecuencias completo se construya correctamente.

    Para una FFT de N muestras tomadas a Fs Hz, NumPy define los bins como:

        f[k] = k * Fs/N

    incluyendo primero las frecuencias no negativas y posteriormente
    las frecuencias negativas correspondientes al espectro bilateral.

    Esta prueba compara TODO el eje frecuencial contra np.fft.fftfreq().
    """

    # Definimos una frecuencia de muestreo de 1000 Hz
    fs = 1000.0

    # Utilizamos N=1024 para trabajar con una potencia de dos
    N = 1024

    # Calculamos la duración necesaria para generar exactamente N muestras
    duration = N / fs

    # Generamos una señal válida para construir el objeto Signal
    signal = generate_sine(
        frequency=10.0,
        sample_rate=fs,
        duration=duration
    )

    # Calculamos nuestra FFT
    complex_vals = compute_fft(signal.samples)

    # Construimos el objeto Spectrum
    spectrum = compute_spectrum(signal, complex_vals)

    # NumPy genera el eje frecuencial teórico esperado para una FFT bilateral
    expected_frequencies = np.fft.fftfreq(
        N,
        d=1.0 / fs
    )

    # El eje de frecuencias debe tener exactamente N posiciones
    assert len(spectrum.frequencies) == N

    # Comparamos TODO el eje contra la definición de referencia de NumPy.
    # Esto verifica tanto las frecuencias positivas como las negativas.
    assert np.allclose(
        spectrum.frequencies,
        expected_frequencies,
        rtol=RTOL,
        atol=ATOL
    )

    # La primera componente debe representar DC, es decir, 0 Hz
    assert np.isclose(
        spectrum.frequencies[0],
        0.0,
        atol=ATOL
    )

    # La separación entre bins debe ser:
    #
    # Delta_f = Fs / N
    expected_delta_f = fs / N

    # Verificamos la resolución frecuencial entre los primeros dos bins
    assert np.isclose(
        spectrum.frequencies[1] - spectrum.frequencies[0],
        expected_delta_f,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_internal_lengths():
    """
    Verifica la consistencia interna del objeto Spectrum.

    Todas las representaciones asociadas a una FFT bilateral deben tener
    exactamente el mismo número de elementos N:

        frecuencias
        valores complejos
        magnitud
        fase
    """

    # Definimos N=256 muestras
    N = 256

    # Utilizamos Fs=N para obtener una duración sencilla de 1 segundo
    fs = 256.0

    # Generamos una senoide válida
    signal = generate_sine(
        frequency=20.0,
        sample_rate=fs,
        duration=1.0
    )

    # Calculamos el espectro complejo
    complex_vals = compute_fft(signal.samples)

    # Ejecutamos nuestro módulo de análisis espectral
    spectrum = compute_spectrum(signal, complex_vals)

    # El eje de frecuencias debe tener N elementos
    assert len(spectrum.frequencies) == N

    # Los valores complejos deben conservar los N bins originales
    assert len(spectrum.complex_values) == N

    # La magnitud debe contener un valor por cada bin
    assert len(spectrum.magnitude) == N

    # La fase debe contener un valor por cada bin
    assert len(spectrum.phase) == N


def test_spectrum_preserves_complex_values():
    """
    Verifica que compute_spectrum() analice el espectro complejo
    sin modificar accidentalmente los valores producidos por la FFT.

    Spectrum debe interpretar X[k], no alterar X[k].
    """

    # Generamos una señal conocida
    signal = generate_sine(
        frequency=32.0,
        sample_rate=256.0,
        duration=1.0
    )

    # Guardamos los valores complejos calculados por nuestra FFT
    complex_vals = compute_fft(signal.samples)

    # Construimos el objeto Spectrum
    spectrum = compute_spectrum(signal, complex_vals)

    # Los valores almacenados en Spectrum deben ser equivalentes
    # a los valores complejos entregados originalmente
    assert np.allclose(
        spectrum.complex_values,
        complex_vals,
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "amplitude",
    [0.5, 1.0, 2.0, 5.0, 10.0]
)
def test_spectrum_sine_magnitude_normalization(amplitude):
    """
    Verifica la normalización de magnitud para diferentes amplitudes.

    Generamos una señal senoidal cuya frecuencia coincide exactamente
    con un bin de la FFT para evitar spectral leakage.

    En el espectro unilateral normalizado esperamos recuperar
    directamente la amplitud física de la senoide.
    """

    # Definimos Fs=1024 Hz
    fs = 1024.0

    # Utilizamos N=1024 muestras
    N = 1024

    # La duración correspondiente es exactamente 1 segundo
    duration = N / fs

    # Elegimos 64 Hz porque:
    #
    # Delta_f = Fs/N = 1 Hz
    #
    # Por tanto, 64 Hz cae exactamente en el bin número 64
    frequency = 64.0

    # Generamos la senoide utilizando la amplitud parametrizada
    signal = generate_sine(
        frequency=frequency,
        sample_rate=fs,
        duration=duration,
        amplitude=amplitude
    )

    # Calculamos nuestra FFT
    complex_vals = compute_fft(signal.samples)

    # Analizamos el espectro
    spectrum = compute_spectrum(signal, complex_vals)

    # Extraemos el espectro unilateral utilizado para interpretar
    # amplitudes físicas de señales reales
    pos_freqs, pos_mags, _ = spectrum.positive_frequencies

    # Localizamos explícitamente el bin correspondiente a +64 Hz
    peak_idx = np.argmin(
        np.abs(pos_freqs - frequency)
    )

    # La magnitud normalizada del pico debe coincidir
    # con la amplitud de la señal temporal
    assert np.isclose(
        pos_mags[peak_idx],
        amplitude,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_dc_normalization():
    """
    Verifica específicamente la normalización de la componente DC.

    Esta prueba es importante porque DC NO debe duplicarse al convertir
    un espectro bilateral en un espectro unilateral.

    Para una señal constante:

        x[n] = 3

    esperamos una componente DC de magnitud exactamente 3.
    """

    # Definimos los parámetros de la señal
    fs = 1024.0
    N = 1024

    # Creamos manualmente una señal constante de amplitud 3
    samples = np.full(
        N,
        3.0
    )

    # Construimos la estructura Signal
    signal = Signal(
        samples=samples,
        sample_rate=fs,
        name="DC_3"
    )

    # Calculamos nuestra FFT
    complex_vals = compute_fft(signal.samples)

    # Analizamos el espectro
    spectrum = compute_spectrum(signal, complex_vals)

    # Extraemos el espectro unilateral
    pos_freqs, pos_mags, _ = spectrum.positive_frequencies

    # La primera frecuencia debe corresponder a DC
    assert np.isclose(
        pos_freqs[0],
        0.0,
        atol=ATOL
    )

    # La magnitud de DC debe ser exactamente la amplitud constante 3
    assert np.isclose(
        pos_mags[0],
        3.0,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_nyquist_normalization():
    """
    Verifica la normalización especial de la frecuencia de Nyquist.

    En un espectro unilateral, el bin de Nyquist no debe duplicarse,
    porque al igual que DC no posee una pareja independiente dentro
    del espectro positivo.

    Para:

        x[n] = A * cos(pi*n)

    la frecuencia corresponde exactamente a Fs/2.
    """

    # Definimos una frecuencia de muestreo de 1024 Hz
    fs = 1024.0

    # Utilizamos un número par de muestras
    N = 1024

    # Definimos una amplitud conocida
    amplitude = 2.0

    # Construimos los índices discretos n
    n = np.arange(N)

    # cos(pi*n) alterna:
    #
    # +1, -1, +1, -1, ...
    #
    # y representa exactamente la frecuencia de Nyquist
    samples = amplitude * np.cos(np.pi * n)

    # Construimos la señal
    signal = Signal(
        samples=samples,
        sample_rate=fs,
        name="Nyquist"
    )

    # Calculamos y analizamos el espectro
    spectrum = compute_spectrum(
        signal,
        compute_fft(signal.samples)
    )

    # Extraemos el espectro unilateral
    pos_freqs, pos_mags, _ = spectrum.positive_frequencies

    # El último bin del espectro unilateral debe corresponder a Fs/2
    assert np.isclose(
        pos_freqs[-1],
        fs / 2,
        rtol=RTOL,
        atol=ATOL
    )

    # La magnitud de Nyquist debe conservar la amplitud original
    # sin ser multiplicada nuevamente por dos
    assert np.isclose(
        pos_mags[-1],
        amplitude,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_phase_extraction():
    """
    Verifica que un desplazamiento de fase en el dominio temporal
    aparezca correctamente en el espectro.

    Para dos señales:

        x1(t) = sin(2*pi*f*t)
        x2(t) = sin(2*pi*f*t + phi)

    la diferencia de fase de la componente espectral positiva debe ser phi.

    Se localiza explícitamente el bin de +f para evitar depender de argmax(),
    ya que una senoide real produce dos picos simétricos en +f y -f.
    """

    # Definimos los parámetros del experimento
    fs = 1024.0
    N = 1024
    duration = N / fs
    frequency = 32.0

    # Definimos un desplazamiento de fase de 45 grados
    phi_shift = np.pi / 4

    # Generamos una primera señal sin desfase
    sig1 = generate_sine(
        frequency=frequency,
        sample_rate=fs,
        duration=duration,
        phase=0.0
    )

    # Calculamos su espectro
    spec1 = compute_spectrum(
        sig1,
        compute_fft(sig1.samples)
    )

    # Generamos una segunda señal con un desfase de pi/4
    sig2 = generate_sine(
        frequency=frequency,
        sample_rate=fs,
        duration=duration,
        phase=phi_shift
    )

    # Calculamos el espectro de la segunda señal
    spec2 = compute_spectrum(
        sig2,
        compute_fft(sig2.samples)
    )

    # Localizamos explícitamente el bin correspondiente a +32 Hz
    peak_idx = np.argmin(
        np.abs(spec1.frequencies - frequency)
    )

    # Calculamos la diferencia angular.
    #
    # Utilizamos exp(j*Delta_phi) y np.angle() para envolver
    # automáticamente la diferencia al intervalo [-pi, pi].
    phase_diff = np.angle(
        np.exp(
            1j * (
                spec2.phase[peak_idx]
                - spec1.phase[peak_idx]
            )
        )
    )

    # La diferencia de fase debe corresponder al desplazamiento aplicado
    assert np.isclose(
        phase_diff,
        phi_shift,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_absolute_phase():
    """
    Verifica la fase absoluta de una senoide alineada con un bin.

    Matemáticamente:

        sin(theta) = cos(theta - pi/2)

    Por lo tanto, para:

        x(t) = A*sin(2*pi*f*t + phi)

    la componente espectral positiva presenta una fase:

        phi - pi/2
    """

    # Definimos parámetros conocidos
    fs = 1024.0
    N = 1024
    frequency = 64.0

    # Utilizamos un desfase conocido
    phi = np.pi / 4

    # Generamos la señal
    signal = generate_sine(
        frequency=frequency,
        sample_rate=fs,
        duration=N / fs,
        phase=phi
    )

    # Calculamos el espectro
    spectrum = compute_spectrum(
        signal,
        compute_fft(signal.samples)
    )

    # Localizamos específicamente el bin de frecuencia positiva
    peak_idx = np.argmin(
        np.abs(spectrum.frequencies - frequency)
    )

    # Calculamos la fase teórica esperada
    expected_phase = phi - np.pi / 2

    # Envolvemos también la fase esperada al intervalo [-pi, pi]
    expected_phase = np.angle(
        np.exp(1j * expected_phase)
    )

    # Comparamos la fase calculada contra la expresión teórica
    assert np.isclose(
        spectrum.phase[peak_idx],
        expected_phase,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_phase_cleaning_with_controlled_values():
    """
    Prueba unitaria directa de la limpieza de fase.

    En lugar de depender del residuo numérico generado por una FFT,
    construimos manualmente un espectro complejo que contiene
    componentes extremadamente pequeñas.

    Cuando la magnitud está por debajo del umbral definido por
    compute_spectrum(), la fase debe limpiarse y establecerse en 0.
    """

    # Creamos una señal auxiliar de N=4 muestras.
    # Las muestras temporales no son relevantes para esta prueba;
    # solamente necesitamos una estructura Signal válida.
    signal = Signal(
        samples=np.zeros(4),
        sample_rate=4.0,
        name="PhaseCleaningTest"
    )

    # Construimos manualmente valores complejos.
    #
    # El primer bin posee magnitud claramente significativa.
    # Los demás contienen únicamente residuos numéricos muy pequeños.
    complex_vals = np.array([
        1.0 + 1.0j,
        1e-15 + 2e-15j,
        -1e-16 + 1e-16j,
        1e-15 - 1e-15j
    ], dtype=np.complex128)

    # Ejecutamos únicamente el módulo que queremos probar
    spectrum = compute_spectrum(
        signal,
        complex_vals
    )

    # La primera componente posee magnitud real y su fase
    # no debería ser eliminada
    assert spectrum.phase[0] != 0.0

    # Las fases correspondientes a magnitudes despreciables
    # deben limpiarse exactamente a cero
    assert np.all(
        spectrum.phase[1:] == 0.0
    )


def test_spectrum_zero_signal():
    """
    Verifica el comportamiento de una señal completamente nula.

    Matemáticamente:

        x[n] = 0
        X[k] = 0

    Por lo tanto:
        magnitud = 0
        fase = 0

    La fase se fuerza a cero porque no tiene significado cuando
    la magnitud de la componente es nula.
    """

    # Definimos una señal de 64 muestras completamente nula
    samples = np.zeros(64)

    # Construimos el objeto Signal
    signal = Signal(
        samples=samples,
        sample_rate=64.0,
        name="ZeroSignal"
    )

    # Calculamos la FFT y posteriormente el espectro
    spectrum = compute_spectrum(
        signal,
        compute_fft(signal.samples)
    )

    # Todas las magnitudes deben ser exactamente cero
    assert np.allclose(
        spectrum.magnitude,
        0.0,
        atol=ATOL
    )

    # Todas las fases deben haber sido limpiadas a cero
    assert np.all(
        spectrum.phase == 0.0
    )


def test_spectrum_multiple_frequency_components():
    """
    Verifica la identificación y normalización de múltiples
    componentes frecuenciales presentes simultáneamente.

    Construimos:

        x(t) =
            2*sin(2*pi*50*t)
            +
            4*sin(2*pi*200*t)

    Debido a que Delta_f = 1 Hz, ambas frecuencias coinciden
    exactamente con bins de la FFT y no debería existir leakage.
    """

    # Definimos una frecuencia de muestreo de 1024 Hz
    fs = 1024.0

    # Utilizamos 1024 muestras para obtener Delta_f = 1 Hz
    N = 1024

    # Construimos el vector temporal
    t = np.arange(N) / fs

    # Generamos una señal formada por dos componentes
    samples = (
        2.0 * np.sin(2 * np.pi * 50.0 * t)
        +
        4.0 * np.sin(2 * np.pi * 200.0 * t)
    )

    # Empaquetamos las muestras en nuestra estructura Signal
    signal = Signal(
        samples=samples,
        sample_rate=fs,
        name="MultiTone"
    )

    # Calculamos y analizamos el espectro
    spectrum = compute_spectrum(
        signal,
        compute_fft(signal.samples)
    )

    # Extraemos únicamente el espectro unilateral
    pos_freqs, pos_mags, _ = spectrum.positive_frequencies

    # Localizamos el bin correspondiente a 50 Hz
    idx_50 = np.argmin(
        np.abs(pos_freqs - 50.0)
    )

    # Localizamos el bin correspondiente a 200 Hz
    idx_200 = np.argmin(
        np.abs(pos_freqs - 200.0)
    )

    # La componente de 50 Hz debe recuperar amplitud 2
    assert np.isclose(
        pos_mags[idx_50],
        2.0,
        rtol=RTOL,
        atol=ATOL
    )

    # La componente de 200 Hz debe recuperar amplitud 4
    assert np.isclose(
        pos_mags[idx_200],
        4.0,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_positive_frequencies():
    """
    Verifica el espectro unilateral utilizado para señales reales.

    Para N par, un espectro unilateral completo debe incluir:

        DC
        frecuencias positivas
        Nyquist

    Por lo tanto, debe contener:

        N/2 + 1

    valores.
    """

    # Definimos una señal con N=1024 muestras
    N = 1024
    fs = 1024.0

    # Generamos una senoide válida
    signal = generate_sine(
        frequency=50.0,
        sample_rate=fs,
        duration=1.0
    )

    # Calculamos el espectro
    spectrum = compute_spectrum(
        signal,
        compute_fft(signal.samples)
    )

    # Extraemos las componentes unilaterales
    pos_freqs, pos_mags, pos_phases = spectrum.positive_frequencies

    # Para N par esperamos N/2 + 1 bins:
    #
    # DC + frecuencias positivas + Nyquist
    assert len(pos_freqs) == N // 2 + 1

    # Magnitudes y fases deben tener exactamente el mismo tamaño
    assert len(pos_mags) == len(pos_freqs)
    assert len(pos_phases) == len(pos_freqs)

    # Todas las frecuencias del espectro unilateral deben ser no negativas
    assert np.all(
        pos_freqs >= 0.0
    )

    # La primera frecuencia debe ser DC
    assert np.isclose(
        pos_freqs[0],
        0.0,
        atol=ATOL
    )

    # La última frecuencia debe representar Nyquist
    assert np.isclose(
        pos_freqs[-1],
        fs / 2,
        rtol=RTOL,
        atol=ATOL
    )


def test_spectrum_fails_on_mismatched_length():
    """
    Verifica que compute_spectrum() rechace un espectro complejo cuya
    longitud no coincida con el número de muestras de la señal.

    No tendría sentido asociar, por ejemplo:

        N señal = 1024
        N espectro = 512

    si estamos trabajando con un espectro bilateral completo.
    """

    # Generamos una señal de 1024 muestras
    signal = generate_sine(
        frequency=50.0,
        sample_rate=1024.0,
        duration=1.0
    )

    # Creamos deliberadamente un espectro complejo de longitud incorrecta
    invalid_complex_values = np.zeros(
        512,
        dtype=np.complex128
    )

    # Esperamos que compute_spectrum() detecte la inconsistencia
    with pytest.raises(ValueError):
        compute_spectrum(
            signal,
            invalid_complex_values
        )


def test_spectrum_fails_on_two_dimensional_values():
    """
    Verifica que compute_spectrum() rechace matrices bidimensionales.

    Nuestro análisis está diseñado para una FFT unidimensional asociada
    a una señal discreta x[n].
    """

    # Creamos una señal válida de cuatro muestras
    signal = Signal(
        samples=np.array([1.0, 0.0, 0.0, 0.0]),
        sample_rate=4.0,
        name="InvalidSpectrumShapeTest"
    )

    # Construimos deliberadamente una matriz 2x2 de valores complejos
    invalid_complex_values = np.array([
        [1 + 0j, 1 + 0j],
        [1 + 0j, 1 + 0j]
    ])

    # Esperamos que la dimensionalidad inválida sea rechazada
    with pytest.raises(ValueError):
        compute_spectrum(
            signal,
            invalid_complex_values
        )

