"""
Módulo: ranging.py
------------------
Responsabilidad: Convertir estimaciones temporales (retardos en el tiempo)
en distancias físicas espaciales expresadas en metros.

Implementa la física básica de propagación de ondas acústicas para un
sistema de radar acústico, incluyendo:

- Conversión de retardos expresados en muestras a segundos.
- Cálculo de la velocidad del sonido según la temperatura ambiente.
- Conversión del tiempo de vuelo del eco a distancia.
- Compensación por el recorrido de ida y vuelta de la señal acústica.
- Validaciones para evitar valores físicos o numéricos inválidos.
"""

import math


# Definimos el valor del cero absoluto en grados Celsius.
# Ninguna temperatura física puede ser igual o inferior a este valor.
ABSOLUTE_ZERO_C = -273.15


def calculate_speed_of_sound(temperature_c: float = 20.0) -> float:
    """
    Calcula la velocidad aproximada del sonido en el aire según la temperatura.

    Utiliza el modelo lineal:
        c = 331.3 + 0.606*T

    donde T representa la temperatura en grados Celsius y c la velocidad
    del sonido en metros por segundo.
    """

    # Convertimos explícitamente la temperatura a float para trabajar
    # siempre con un valor numérico compatible con los cálculos posteriores.
    temperature_c = float(temperature_c)

    # Verificamos que la temperatura sea un número finito.
    # Esto permite rechazar valores especiales como NaN, +infinito o -infinito.
    if not math.isfinite(temperature_c):

        # Si el valor no representa una temperatura numérica válida,
        # detenemos el cálculo mediante una excepción.
        raise ValueError(
            "La temperatura debe ser un valor finito."
        )

    # Verificamos que la temperatura sea físicamente posible.
    # El cero absoluto (-273.15 °C) representa el límite inferior físico.
    if temperature_c <= ABSOLUTE_ZERO_C:

        # Una temperatura igual o inferior al cero absoluto no puede
        # utilizarse dentro del modelo físico del sistema.
        raise ValueError(
            "La temperatura debe ser superior al cero absoluto "
            "(-273.15 °C)."
        )

    # La velocidad del sonido a 0 °C se aproxima a 331.3 m/s.
    # Por cada grado Celsius de aumento, la velocidad aumenta
    # aproximadamente 0.606 m/s.
    speed = 331.3 + (0.606 * temperature_c)

    # Retornamos la velocidad calculada en metros por segundo (m/s).
    return speed


def calculate_distance(
    delay_seconds: float,
    speed_of_sound: float = 343.0
) -> float:
    """
    Convierte el tiempo de vuelo de un eco en la distancia física al objeto.

    Utiliza la relación:
        distancia = (velocidad * tiempo) / 2

    La división entre dos compensa el recorrido de ida y vuelta de la señal.
    """

    # Convertimos el retardo a float para garantizar que pueda utilizarse
    # correctamente en las operaciones matemáticas posteriores.
    delay_seconds = float(delay_seconds)

    # Convertimos también la velocidad del sonido a float para mantener
    # consistencia numérica durante el cálculo.
    speed_of_sound = float(speed_of_sound)

    # Verificamos que el tiempo de retardo sea un número finito.
    # Valores como NaN o infinito no representan tiempos físicos válidos.
    if not math.isfinite(delay_seconds):

        # Rechazamos el valor antes de continuar con el cálculo de distancia.
        raise ValueError(
            "El tiempo de retardo debe ser un valor finito."
        )

    # Verificamos que la velocidad del sonido también sea un número finito.
    # Una velocidad infinita o indefinida no tiene significado físico.
    if not math.isfinite(speed_of_sound):

        # Detenemos el cálculo si la velocidad recibida es inválida.
        raise ValueError(
            "La velocidad del sonido debe ser un valor finito."
        )

    # En un sistema físico, un retardo negativo significaría que el eco
    # fue recibido antes de transmitir la señal, lo cual viola la causalidad.
    if delay_seconds < 0.0:

        # Rechazamos cualquier tiempo de vuelo negativo.
        raise ValueError(
            "El tiempo de retardo no puede ser negativo."
        )

    # La velocidad de propagación debe ser estrictamente positiva.
    # Una velocidad igual a cero o negativa no permite representar
    # correctamente la propagación de una onda acústica.
    if speed_of_sound <= 0.0:

        # Detenemos el cálculo si la velocidad no es físicamente válida.
        raise ValueError(
            "La velocidad del sonido debe ser estrictamente positiva."
        )

    # Calculamos la distancia total recorrida por la onda acústica.
    # Este valor representa la distancia de ida hasta el objeto
    # más la distancia de regreso hasta el receptor.
    total_distance = speed_of_sound * delay_seconds

    # Dividimos la distancia total entre dos porque el tiempo medido
    # contiene el recorrido completo:
    #
    # Transmisor (TX) -> Objeto -> Receptor (RX)
    #
    # El resultado representa únicamente la distancia entre
    # el sistema de radar y el objeto.
    distance_to_object = total_distance / 2.0

    # Retornamos la distancia estimada al objeto en metros.
    return distance_to_object


def estimate_range(
    delay_samples: float,
    sample_rate: float,
    temperature_c: float = 20.0
) -> float:
    """
    Convierte directamente un retardo expresado en muestras a una distancia.

    Esta función coordina el flujo completo:
        muestras -> segundos -> velocidad del sonido -> distancia.

    También permite recibir retardos fraccionarios para soportar futuras
    técnicas de estimación submuestra.
    """

    # Convertimos el retardo en muestras a float.
    # Esto permite trabajar tanto con retardos enteros como con posibles
    # retardos fraccionarios obtenidos mediante interpolación.
    delay_samples = float(delay_samples)

    # Convertimos la frecuencia de muestreo a float para garantizar
    # consistencia en la división posterior.
    sample_rate = float(sample_rate)

    # Verificamos que el número de muestras de retardo sea finito.
    # NaN o infinito no pueden representar una posición válida
    # dentro de una señal digital.
    if not math.isfinite(delay_samples):

        # Rechazamos cualquier valor numéricamente inválido.
        raise ValueError(
            "El retardo en muestras debe ser un valor finito."
        )

    # Un retardo negativo implicaría que el eco fue recibido
    # antes del instante de transmisión.
    if delay_samples < 0.0:

        # Rechazamos retardos que violen la causalidad del sistema.
        raise ValueError(
            "El retardo en muestras no puede ser negativo."
        )

    # Verificamos que la frecuencia de muestreo sea un valor finito.
    # Una frecuencia NaN o infinita no tiene significado dentro
    # de un sistema de adquisición digital.
    if not math.isfinite(sample_rate):

        # Detenemos el procesamiento si Fs no es un valor válido.
        raise ValueError(
            "La frecuencia de muestreo debe ser un valor finito."
        )

    # La frecuencia de muestreo debe ser estrictamente mayor que cero.
    # Un valor igual a cero causaría una división entre cero y
    # una frecuencia negativa no tiene sentido físico.
    if sample_rate <= 0.0:

        # Rechazamos cualquier frecuencia de muestreo inválida.
        raise ValueError(
            "La frecuencia de muestreo debe ser mayor a cero."
        )

    # 1. Convertimos el retardo desde el dominio discreto
    #    (cantidad de muestras) hacia tiempo continuo en segundos.
    #
    #    tiempo = muestras / frecuencia_de_muestreo
    delay_seconds = delay_samples / sample_rate

    # 2. Calculamos la velocidad del sonido correspondiente
    #    a la temperatura ambiente proporcionada.
    #
    #    Esto permite compensar parcialmente el efecto térmico
    #    sobre la propagación de la onda acústica.
    speed = calculate_speed_of_sound(temperature_c)

    # 3. Convertimos el tiempo de vuelo del eco en distancia física.
    #    La función calculate_distance() aplica automáticamente
    #    la compensación del recorrido de ida y vuelta.
    distance = calculate_distance(
        delay_seconds,
        speed_of_sound=speed
    )

    # Retornamos la distancia final estimada al objeto en metros.
    return distance