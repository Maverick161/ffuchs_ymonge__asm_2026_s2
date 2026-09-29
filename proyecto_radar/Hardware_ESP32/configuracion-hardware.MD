# Especificación y Configuración del Hardware: Radar Acústico ToF

Este documento resume la arquitectura física, electrónica y acústica del prototipo de radar acústico desarrollado sobre el microcontrolador ESP32-S3. Sirve como referencia técnica de contexto para el desarrollo, depuración y calibración del sistema.

---

## 1. Componentes Principales y Arquitectura Electrónica

| Componente | Modelo / Referencia | Función en el Sistema | Conexión / Pines |
| :--- | :--- | :--- | :--- |
| **Microcontrolador** | **ESP32-S3** (Dual-Core LX7 @ 240 MHz) | Generación del chirp (Core 0) y muestreo ADC + procesamiento FFT/ToF (Core 1) | `GPIO 4` (TX PWM), `GPIO 1` (RX ADC), `GPIO 8` (SDA), `GPIO 9` (SCL) |
| **Sensor Acústico** | **KY-038** (Cápsula electret) | Captura de ondas de presión acústica del eco reflejado | Salida analógica hacia el preamplificador |
| **Preamplificador** | Discreto con **BC549 NPN** | Preamplificación de bajo ruido y acople de nivel DC hacia el ADC de 12 bits | Entrada desde KY-038, salida a `GPIO 1` |
| **Amplificador TX** | **TPA3110** (Clase D) | Amplificación de potencia del chirp lineal (2 kHz a 8 kHz) | Entrada desde `GPIO 4`, alimentación 12V |
| **Emisor (Buffer)** | Parlante de **8 $\Omega$** | Transducción electroacústica para emisión del pulso acústico | Salida del TPA3110 |
| **Pantalla** | **LCD 16x2** con I2C (**PCF8574**) | Visualización en tiempo real de distancia (cm) y estado | `SDA: GPIO 8`, `SCL: GPIO 9` (Dirección `0x3F`) |

---

## 2. Circuito Preamplificador de Bajo Ruido (Transistor BC549)

Para garantizar un rango dinámico óptimo en el ADC del ESP32-S3 sin pérdida de señal en ecos lejanos, se diseñó e implementó un preamplificador en configuración de emisor común con transistor BJT NPN **BC549** (grado de bajo ruido de audio), alimentado directamente desde los 3.3 V del microcontrolador:

### Parámetros y Valores del Esquemático (LTspice)
* **Transistor:** Q1 (`BC549` NPN, $h_{fe} \approx 400$).
* **Polarización de Base:** Divisor de tensión formado por $R_{b2} = 68\text{ k}\Omega$ (pull-up a 3.3 V) y $R_{b1} = 22\text{ k}\Omega$ (pull-down a GND), estableciendo un punto de polarización DC en la base de $\approx 0.8\text{ V}$.
* **Acople de Entrada AC:** Capacitor $C_1 = 0.1\ \mu\text{F}$ desde la salida del módulo KY-038 a la base de Q1, bloqueando el nivel continuo del micrófono electret.
* **Carga de Colector:** $R_c = 4.7\text{ k}\Omega$ conectado a 3.3 V. La señal de salida se toma directamente del colector hacia el `GPIO 1` del ESP32-S3, oscilando alrededor de un nivel DC centrado para aprovechar el rango del convertidor analógico-digital de 12 bits.
* **Estabilidad y Ganancia de Emisor:**
  * Resistencia degenerativa en serie: $R_{e1} = 47\ \Omega$.
  * Resistencia de polarización DC: $R_{e2} = 330\ \Omega$.
  * Capacitor de paso de emisor: $C_e = 10\ \mu\text{F}$ en paralelo con $R_{e2}$.
  * **Ganancia en corriente alterna (AC):** Fijada por la relación $A_v \approx -R_c / R_{e1} = -4700 / 47 \approx -100$ ($\approx 40\text{ dB}$ de amplificación de voltaje en la banda del chirp).
* **Filtrado y Desacople de Alimentación:** Capacitores $C_3 = 100\ \mu\text{F}$ (electrolítico) y $C_2 = 0.1\ \mu\text{F}$ (cerámico) en paralelo entre la línea de 3.3 V y GND para suprimir rizado y ruido de conmutación del microcontrolador.

```
       +3.3V
         |
    +----+-----------+---------------+
    |    |           |               |
   [C3] [C2]       [Rb2]           [Rc]
   100u  0.1u       68k            4.7k
    |    |           |               |
    |    |           +-------+       +-----> GPIO 1 (ADC ESP32)
    |    |           |       |       | (C)
   GND  GND          |     [C1]    --+-- Q1 (BC549)
                     |     0.1u   / (B) \ (E)
          KY-038 ----+-------+           |
          (Audio IN)         |          [Re1] 47 ohm
                           [Rb1]         |
                            22k          +-------+
                             |           |       |
                            GND        [Re2]    [Ce]
                                        330     10u
                                         |       |
                                        GND     GND
```

---

## 3. Disposición Mecánica y Geometría del Setup

El montaje físico fue calibrado para mitigar acoples acústicos directos (*crosstalk*) y reflexiones parásitas en la superficie de prueba:

1. **Alineación de Transductores (Eje X / Eje Y):**
   * El micrófono electret y el parlante (buffer de 8 $\Omega$) están ubicados en la misma coordenada horizontal ($X$).
   * En el eje vertical ($Y$), el micrófono está montado a una altura de **7 a 8 cm directamente por encima** del parlante.
2. **Aislamiento Mecánico contra Vibraciones Estructurales:**
   * El micrófono está montado sobre una cama de desacople hecha de **espuma de lavar platos (poliuretano de celda abierta)**.
   * Esto previene que las vibraciones mecánicas de la membrana del parlante se transmitan a través del chasis plástico/madera directamente hacia la cápsula del micrófono.
3. **Bafle Acústico / Visera Anti-Acople Frontal:**
   * Al frente del micrófono se construyó una **visera de espuma de 2 cm de grosor por 6 cm de largo**.
   * Esta visera sobresale hacia el frente entre el parlante y el micrófono, actuando como una barrera acústica absorbente. Atenúa drásticamente la onda directa emitida por el parlante hacia el micrófono antes de que viaje hacia adelante, garantizando que el receptor capte con máxima pureza el eco reflejado proveniente del blanco.
4. **Inclinación del Chasis del Radar:**
   * La estructura del radar cuenta con una **leve inclinación frontal hacia arriba de aproximadamente 0.8 cm**.
   * Dicha inclinación evita que el haz acústico principal golpee de forma rasante la mesa de pruebas a distancias cortas (30 cm a 45 cm), reduciendo los rebotes parásitos de primer orden en la madera o formica.

---

## 4. Blanco de Prueba (Reflector)

* **Dimensiones:** Disco circular de **36 cm de diámetro**.
* **Material:** Recubierto completamente con **papel de aluminio**. El aluminio ofrece una impedancia acústica muy alta en comparación con el aire ($Z_{aluminio} \gg Z_{aire}$), garantizando un coeficiente de reflexión cercano al 100% ($R \approx 1$) para ondas acústicas en el rango de 2 kHz a 8 kHz.
* **Distancias nominales de ensayo y calibración:**
  * **30 cm**
  * **35 cm**
  * **40 cm**
  * **45 cm**
* **Mesa de prueba:** Despejada durante el encendido inicial para la sustracción de fondo estático (`BACKGROUND_SCALE = 0.98f`).
