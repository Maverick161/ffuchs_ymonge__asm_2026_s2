"""
Módulo: test_echoes.py
----------------------
Responsabilidad: Validar la generación de señales chirp y la simulación del
entorno acústico.

Las pruebas verifican:
- Propiedades matemáticas del chirp.
- Retardos conocidos.
- Atenuación.
- Superposición lineal de múltiples ecos.
- Conservación de frecuencia de muestreo.
- Inmutabilidad de la señal transmitida.
- Manejo de ruido.
- Reproducibilidad.
- Casos límite y configuraciones inválidas.
- Integración entre generación de chirp y simulación de ecos.

Para ejecutar:
pytest tests/test_echoes.py
"""

import numpy as np
import pytest

from radar.signals.generators import generate_chirp
from radar.simulation.echoes import EchoConfig, simulate_echoes
from radar.signals.signal import Signal


# Definimos tolerancias numéricas explícitas para las comparaciones
# de punto flotante utilizadas durante las pruebas.
RTOL = 1e-10
ATOL = 1e-12


def test_generate_chirp_properties():
    """
    Verifica las propiedades estructurales básicas del chirp:
    número de muestras, frecuencia de muestreo, duración y vector temporal.
    """

    # Definimos una frecuencia de muestreo de 1000 muestras por segundo
    fs = 1000.0

    # Definimos una duración de 2 segundos
    duration = 2.0

    # Generamos un chirp ascendente de 10 Hz a 100 Hz
    chirp = generate_chirp(
        start_frequency=10.0,
        end_frequency=100.0,
        sample_rate=fs,
        duration=duration
    )

    # N = Fs * T = 1000 * 2 = 2000 muestras
    assert chirp.number_of_samples == 2000

    # La frecuencia de muestreo debe conservarse exactamente
    assert chirp.sample_rate == fs

    # La duración debe coincidir con la solicitada
    assert np.isclose(
        chirp.duration,
        duration,
        rtol=RTOL,
        atol=ATOL
    )

    # La primera muestra debe ocurrir en t = 0
    assert np.isclose(
        chirp.time_vector[0],
        0.0,
        atol=ATOL
    )

    # Para una fase inicial de cero, una implementación basada en seno
    # debe iniciar en amplitud cero.
    assert np.isclose(
        chirp.samples[0],
        0.0,
        atol=ATOL
    )


def test_generate_chirp_matches_linear_chirp_equation():
    """
    Verifica que las muestras generadas correspondan realmente
    a la ecuación matemática de un chirp lineal.

    Para un chirp cuya frecuencia cambia linealmente:

        f(t) = f0 + k*t

    donde:

        k = (f1 - f0) / T

    la fase acumulada es:

        phi(t) = 2*pi*(f0*t + 0.5*k*t^2) + phi0

    y la señal:

        x(t) = A*sin(phi(t))
    """

    # Definimos parámetros conocidos
    f0 = 20.0
    f1 = 120.0
    fs = 1000.0
    duration = 1.0
    amplitude = 2.0
    phase = np.pi / 6

    # Generamos el chirp mediante nuestra implementación
    chirp = generate_chirp(
        start_frequency=f0,
        end_frequency=f1,
        sample_rate=fs,
        duration=duration,
        amplitude=amplitude,
        phase=phase
    )

    # Construimos el vector temporal independiente
    t = np.arange(chirp.number_of_samples) / fs

    # Calculamos la pendiente de frecuencia del chirp
    k = (f1 - f0) / duration

    # Calculamos la fase teórica en cada instante
    expected_phase = (
        2 * np.pi * (
            f0 * t
            + 0.5 * k * t**2
        )
        + phase
    )

    # Evaluamos la señal teórica esperada
    expected_samples = amplitude * np.sin(expected_phase)

    # Verificamos que la implementación coincida con la ecuación
    assert np.allclose(
        chirp.samples,
        expected_samples,
        rtol=RTOL,
        atol=ATOL
    )


def test_generate_chirp_amplitude_and_phase():
    """
    Verifica específicamente el manejo de amplitud y fase inicial.

    Para t = 0:

        x(0) = A*sin(phi)

    Si phi = pi/2:

        x(0) = A
    """

    # Definimos una amplitud conocida
    amplitude = 3.0

    # Utilizamos pi/2 para obtener sin(pi/2) = 1
    phase = np.pi / 2

    # Generamos el chirp
    chirp = generate_chirp(
        start_frequency=10.0,
        end_frequency=100.0,
        sample_rate=1000.0,
        duration=1.0,
        amplitude=amplitude,
        phase=phase
    )

    # La primera muestra debe coincidir con la amplitud
    assert np.isclose(
        chirp.samples[0],
        amplitude,
        rtol=RTOL,
        atol=ATOL
    )


def test_generate_chirp_descending():
    """
    Verifica que el generador soporte chirps descendentes.

    Un chirp puede cambiar desde una frecuencia alta hacia
    una frecuencia más baja sin representar una entrada inválida.
    """

    # Generamos un chirp descendente de 300 Hz a 50 Hz
    chirp = generate_chirp(
        start_frequency=300.0,
        end_frequency=50.0,
        sample_rate=1000.0,
        duration=1.0
    )

    # La señal debe crearse correctamente
    assert chirp.number_of_samples == 1000

    # Las muestras deben ser finitas
    assert np.all(
        np.isfinite(chirp.samples)
    )


@pytest.mark.parametrize(
    "sample_rate, duration",
    [
        (0.0, 1.0),
        (-1000.0, 1.0),
        (1000.0, 0.0),
        (1000.0, -1.0),
    ]
)
def test_generate_chirp_fails_on_invalid_sampling_parameters(
    sample_rate,
    duration
):
    """
    Verifica que el generador rechace frecuencias de muestreo
    y duraciones inválidas.

    Para el sistema:

        Fs > 0
        T > 0
    """

    # Cualquier combinación inválida debe provocar ValueError
    with pytest.raises(ValueError):
        generate_chirp(
            start_frequency=10.0,
            end_frequency=100.0,
            sample_rate=sample_rate,
            duration=duration
        )


def test_simulate_echoes_single_clean():
    """
    Verifica que un único eco se desplace exactamente la cantidad
    de muestras solicitada y se atenúe correctamente.
    """

    # Creamos una señal TX sencilla en forma de impulso
    tx = Signal(
        samples=np.array([1.0, 0.0, 0.0]),
        sample_rate=1.0,
        name="Impulse"
    )

    # Configuramos un eco retardado 2 muestras
    # con amplitud reducida a la mitad
    echo_cfg = EchoConfig(
        delay_samples=2,
        attenuation=0.5
    )

    # Simulamos la señal recibida
    rx = simulate_echoes(
        tx,
        [echo_cfg]
    )

    # La longitud esperada es:
    #
    # N_rx = N_tx + max_delay
    #
    # 3 + 2 = 5
    assert rx.number_of_samples == 5

    # El impulso debe aparecer exactamente en el índice 2
    # con amplitud 0.5
    expected_rx = np.array([
        0.0,
        0.0,
        0.5,
        0.0,
        0.0
    ])

    # Comparamos toda la señal recibida
    assert np.allclose(
        rx.samples,
        expected_rx,
        rtol=RTOL,
        atol=ATOL
    )

    # La simulación de ecos no debe cambiar Fs
    assert rx.sample_rate == tx.sample_rate


def test_simulate_echoes_zero_delay():
    """
    Verifica el caso frontera delay = 0.

    Si el eco no posee retardo:

        RX[n] = attenuation * TX[n]
    """

    # Creamos una señal TX conocida
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate=1000.0,
        name="ZeroDelayTest"
    )

    # Configuramos un eco sin retardo
    echo_cfg = EchoConfig(
        delay_samples=0,
        attenuation=0.5
    )

    # Simulamos el eco
    rx = simulate_echoes(
        tx,
        [echo_cfg]
    )

    # El resultado esperado es simplemente TX multiplicada por 0.5
    expected_rx = np.array([
        0.5,
        1.0,
        1.5
    ])

    # Comprobamos el resultado
    assert np.allclose(
        rx.samples,
        expected_rx,
        rtol=RTOL,
        atol=ATOL
    )


def test_simulate_echoes_zero_attenuation():
    """
    Verifica que una atenuación igual a cero sea válida.

    Matemáticamente:

        0 * x[n-d] = 0

    Por lo tanto, el eco no debe contribuir amplitud a la señal recibida.
    """

    # Creamos una señal conocida
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate=1.0
    )

    # Configuramos un eco completamente atenuado
    echo_cfg = EchoConfig(
        delay_samples=2,
        attenuation=0.0
    )

    # Ejecutamos la simulación
    rx = simulate_echoes(
        tx,
        [echo_cfg]
    )

    # La longitud debe incluir el retardo
    assert rx.number_of_samples == 5

    # Todas las amplitudes deben permanecer en cero
    assert np.allclose(
        rx.samples,
        0.0,
        atol=ATOL
    )


def test_simulate_echoes_multiple_overlap():
    """
    Verifica la superposición lineal de múltiples ecos que
    coinciden parcialmente en el tiempo.

    El objetivo es comprobar que los ecos sean SUMADOS
    y no sobrescritos.
    """

    # Señal TX sencilla de dos muestras
    tx = Signal(
        samples=np.array([1.0, 1.0]),
        sample_rate=1.0,
        name="Pulse"
    )

    # Creamos dos ecos que se superponen parcialmente
    echoes = [
        EchoConfig(
            delay_samples=0,
            attenuation=1.0
        ),
        EchoConfig(
            delay_samples=1,
            attenuation=0.5
        )
    ]

    # Simulamos los ecos
    rx = simulate_echoes(
        tx,
        echoes
    )

    # Desarrollo esperado:
    #
    # idx 0:
    # 1.0 * 1.0 = 1.0
    #
    # idx 1:
    # 1.0 * 1.0 + 1.0 * 0.5 = 1.5
    #
    # idx 2:
    # 1.0 * 0.5 = 0.5
    expected_rx = np.array([
        1.0,
        1.5,
        0.5
    ])

    # Verificamos la superposición lineal
    assert np.allclose(
        rx.samples,
        expected_rx,
        rtol=RTOL,
        atol=ATOL
    )


def test_simulate_echoes_longest_delay_determines_length():
    """
    Verifica que la longitud de la señal recibida esté determinada
    por el eco con mayor retardo.

    Debe cumplirse:

        N_rx = N_tx + max(delay_samples)
    """

    # TX de longitud 4
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_rate=1.0
    )

    # Deliberadamente colocamos los retardos en un orden no creciente
    echoes = [
        EchoConfig(delay_samples=2, attenuation=1.0),
        EchoConfig(delay_samples=10, attenuation=0.5),
        EchoConfig(delay_samples=7, attenuation=0.25)
    ]

    # Ejecutamos la simulación
    rx = simulate_echoes(
        tx,
        echoes
    )

    # N_tx = 4
    # max_delay = 10
    # N_rx = 14
    assert rx.number_of_samples == 14


def test_simulate_echoes_order_does_not_matter():
    """
    Verifica la propiedad conmutativa de la suma de ecos.

    El resultado físico no debe depender del orden en el que
    las configuraciones sean entregadas al simulador.
    """

    # Creamos una señal TX conocida
    tx = Signal(
        samples=np.array([1.0, 2.0, 1.0]),
        sample_rate=1.0
    )

    # Definimos tres ecos
    echo1 = EchoConfig(
        delay_samples=0,
        attenuation=1.0
    )

    echo2 = EchoConfig(
        delay_samples=2,
        attenuation=0.5
    )

    echo3 = EchoConfig(
        delay_samples=5,
        attenuation=0.25
    )

    # Primera simulación en un orden
    rx_a = simulate_echoes(
        tx,
        [echo1, echo2, echo3]
    )

    # Segunda simulación utilizando un orden diferente
    rx_b = simulate_echoes(
        tx,
        [echo3, echo1, echo2]
    )

    # La suma de ecos debe producir exactamente el mismo resultado
    assert np.allclose(
        rx_a.samples,
        rx_b.samples,
        rtol=RTOL,
        atol=ATOL
    )


def test_simulate_echoes_does_not_modify_tx():
    """
    Verifica una propiedad arquitectónica importante:
    simulate_echoes() debe crear una nueva señal RX y no modificar
    las muestras de la señal transmitida original.
    """

    # Creamos la señal TX
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0, 4.0]),
        sample_rate=1000.0
    )

    # Guardamos una copia independiente antes de la simulación
    original_samples = tx.samples.copy()

    # Ejecutamos una simulación cualquiera
    simulate_echoes(
        tx,
        [
            EchoConfig(
                delay_samples=3,
                attenuation=0.5
            )
        ]
    )

    # Las muestras originales deben permanecer intactas
    assert np.array_equal(
        tx.samples,
        original_samples
    )


def test_simulate_echoes_preserves_sample_rate():
    """
    Verifica explícitamente que la simulación preserve
    la frecuencia de muestreo de la señal original.

    Esto es crítico porque posteriormente:

        delay_seconds = delay_samples / Fs
    """

    # Generamos una señal con Fs representativa de audio
    tx = Signal(
        samples=np.array([1.0, 0.0, 0.0]),
        sample_rate=48000.0
    )

    # Simulamos un eco
    rx = simulate_echoes(
        tx,
        [
            EchoConfig(
                delay_samples=100,
                attenuation=0.8
            )
        ]
    )

    # Fs debe permanecer exactamente igual
    assert rx.sample_rate == 48000.0


def test_simulate_echoes_with_chirp():
    """
    Prueba de integración entre generate_chirp() y simulate_echoes().

    Se genera un chirp real, se aplica un retardo conocido y se
    verifica que las muestras recibidas correspondan exactamente
    a una versión desplazada y atenuada de TX.
    """

    # Generamos una señal chirp de prueba
    tx = generate_chirp(
        start_frequency=50.0,
        end_frequency=200.0,
        sample_rate=1000.0,
        duration=0.1
    )

    # Definimos un retardo conocido
    delay = 25

    # Definimos una atenuación conocida
    attenuation = 0.6

    # Simulamos el eco
    rx = simulate_echoes(
        tx,
        [
            EchoConfig(
                delay_samples=delay,
                attenuation=attenuation
            )
        ]
    )

    # Antes del eco deben existir exactamente "delay" ceros
    assert np.allclose(
        rx.samples[:delay],
        0.0,
        atol=ATOL
    )

    # Desde el índice del retardo debe aparecer la señal TX
    # multiplicada por la atenuación configurada
    assert np.allclose(
        rx.samples[
            delay:delay + tx.number_of_samples
        ],
        attenuation * tx.samples,
        rtol=RTOL,
        atol=ATOL
    )


def test_simulate_echoes_zero_noise_variance():
    """
    Verifica que una varianza de ruido igual a cero produzca
    exactamente la misma señal que una simulación limpia.

    Si:

        noise_variance = 0

    entonces:

        w[n] = 0
    """

    # Creamos una señal TX conocida
    tx = Signal(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate=1000.0
    )

    # Configuramos un eco simple
    echoes = [
        EchoConfig(
            delay_samples=2,
            attenuation=0.5
        )
    ]

    # Ejecutamos la simulación sin ruido
    clean_rx = simulate_echoes(
        tx,
        echoes
    )

    # Creamos un generador pseudoaleatorio
    rng = np.random.default_rng(42)

    # Ejecutamos la misma simulación solicitando varianza cero
    zero_noise_rx = simulate_echoes(
        tx,
        echoes,
        noise_variance=0.0,
        rng=rng
    )

    # Ambas señales deben coincidir
    assert np.allclose(
        clean_rx.samples,
        zero_noise_rx.samples,
        rtol=RTOL,
        atol=ATOL
    )


def test_simulate_echoes_noise_is_reproducible():
    """
    Verifica que utilizar exactamente la misma semilla produzca
    la misma realización de ruido.

    Esto es importante para reproducir experimentos,
    gráficas y resultados del informe.
    """

    # Generamos un chirp de prueba
    tx = generate_chirp(
        start_frequency=50.0,
        end_frequency=200.0,
        sample_rate=1000.0,
        duration=0.1
    )

    # Configuramos un eco
    echoes = [
        EchoConfig(
            delay_samples=20,
            attenuation=0.8
        )
    ]

    # Creamos dos generadores independientes con la misma semilla
    rng1 = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)

    # Simulamos la primera señal con ruido
    rx1 = simulate_echoes(
        tx,
        echoes,
        noise_variance=0.1,
        rng=rng1
    )

    # Simulamos nuevamente con la misma semilla
    rx2 = simulate_echoes(
        tx,
        echoes,
        noise_variance=0.1,
        rng=rng2
    )

    # Ambas realizaciones deben ser idénticas
    assert np.array_equal(
        rx1.samples,
        rx2.samples
    )


def test_simulate_echoes_noise_changes_received_signal():
    """
    Verifica la integración del ruido dentro del simulador.

    Una señal recibida con ruido de varianza positiva debe
    diferir de la señal recibida limpia.
    """

    # Generamos una señal TX suficientemente larga
    tx = generate_chirp(
        start_frequency=20.0,
        end_frequency=200.0,
        sample_rate=2000.0,
        duration=0.5
    )

    # Definimos un eco
    echoes = [
        EchoConfig(
            delay_samples=50,
            attenuation=0.7
        )
    ]

    # Obtenemos la señal limpia
    clean_rx = simulate_echoes(
        tx,
        echoes
    )

    # Creamos un generador reproducible
    rng = np.random.default_rng(42)

    # Generamos una señal con ruido
    noisy_rx = simulate_echoes(
        tx,
        echoes,
        noise_variance=0.2,
        rng=rng
    )

    # Las señales deben tener exactamente el mismo tamaño
    assert (
        noisy_rx.number_of_samples
        == clean_rx.number_of_samples
    )

    # La introducción de ruido debe modificar las muestras
    assert not np.allclose(
        noisy_rx.samples,
        clean_rx.samples
    )


def test_simulate_echoes_fails_on_negative_noise_variance():
    """
    Verifica que una varianza negativa sea rechazada.

    Matemáticamente:

        sigma² >= 0
    """

    # Creamos una señal TX válida
    tx = Signal(
        samples=np.array([1.0, 0.0]),
        sample_rate=1.0
    )

    # Creamos un eco válido
    echoes = [
        EchoConfig(
            delay_samples=1,
            attenuation=1.0
        )
    ]

    # Una varianza negativa no es válida
    with pytest.raises(ValueError):
        simulate_echoes(
            tx,
            echoes,
            noise_variance=-0.1
        )


def test_simulate_echoes_fails_on_empty_echo_list():
    """
    Verifica el contrato definido para esta implementación:
    debe existir al menos un eco configurado.

    Una lista vacía se considera inválida dentro del modelo
    adoptado para esta etapa.
    """

    # Creamos una señal TX válida
    tx = Signal(
        samples=np.array([1.0]),
        sample_rate=1.0
    )

    # Esperamos que una lista vacía sea rechazada
    with pytest.raises(ValueError):
        simulate_echoes(
            tx,
            []
        )


def test_simulate_echoes_fails_on_negative_delay():
    """
    Verifica que un retardo negativo sea rechazado.

    En nuestro modelo acústico un eco no puede aparecer
    antes de la señal transmitida.
    """

    # Creamos una señal TX válida
    tx = Signal(
        samples=np.array([1.0]),
        sample_rate=1.0
    )

    # Configuramos deliberadamente un retardo inválido
    invalid_echo = EchoConfig(
        delay_samples=-10,
        attenuation=1.0
    )

    # El simulador debe rechazar la configuración
    with pytest.raises(ValueError):
        simulate_echoes(
            tx,
            [invalid_echo]
        )


def test_simulate_echoes_fails_on_negative_attenuation():
    """
    Verifica el contrato adoptado para la atenuación.

    En esta etapa definimos attenuation como un factor
    de escala no negativo:

        attenuation >= 0

    Por lo tanto, una atenuación negativa se considera
    una configuración no permitida por nuestro modelo.
    """

    # Creamos una señal válida
    tx = Signal(
        samples=np.array([1.0]),
        sample_rate=1.0
    )

    # Definimos una atenuación inválida según nuestro contrato
    invalid_echo = EchoConfig(
        delay_samples=10,
        attenuation=-0.5
    )

    # Esperamos una excepción controlada
    with pytest.raises(ValueError):
        simulate_echoes(
            tx,
            [invalid_echo]
        )


def test_simulate_echoes_fails_on_fractional_delay():
    """
    Verifica que los retardos fraccionarios sean rechazados.

    En esta etapa solamente modelamos retardos discretos enteros.

    Por ejemplo:

        delay_samples = 2.5

    requeriría técnicas adicionales de interpolación o filtros
    de fractional delay, las cuales están fuera del alcance actual.
    """

    # Creamos una señal TX válida
    tx = Signal(
        samples=np.array([1.0, 2.0]),
        sample_rate=1.0
    )

    # Intentamos crear un eco con retardo no entero
    invalid_echo = EchoConfig(
        delay_samples=2.5,
        attenuation=1.0
    )

    # El simulador debe rechazar este tipo de retardo
    with pytest.raises((ValueError, TypeError)):
        simulate_echoes(
            tx,
            [invalid_echo]
        )