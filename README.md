# Sistema de radar acústico

Proyecto de Procesamiento Digital de Señales para estimar distancias mediante una señal acústica conocida, simulación de ecos, correlación y análisis espectral.

El proyecto abarca tanto el modelado matemático y simulación en Python (comparando DFT directa con una FFT radix-2 propia, aliasing, fuga espectral, zero-padding, ruido y estimación de tiempo de vuelo ToF) como su implementación física en tiempo real sobre un microcontrolador ESP32-S3 con hardware dedicado de transmisión, preamplificación de bajo ruido y desacoplo acústico.

## Requisitos

- Python 3.10 o superior. El proyecto fue probado con Python 3.13.
- `pip` actualizado.
- Sistema operativo Windows, Linux o macOS.

## Instalación

Desde la raíz del repositorio, entrar al directorio de la simulación:

```bash
cd proyecto_radar/simulacion_python
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Para instalar el paquete `radar` en modo editable, de forma que sus módulos puedan importarse desde cualquier ubicación del entorno activo:

```bash
python -m pip install -e .
```

Las dependencias principales son:

- `numpy`: arreglos, operaciones numéricas y generación de ruido.
- `scipy`: soporte científico del entorno del proyecto.
- `matplotlib`: gráficas y figuras de los experimentos.
- `pytest`: ejecución de las pruebas automatizadas.

Se recomienda utilizar un entorno virtual:

```bash
python -m venv .venv
```

Activación en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activación en Linux o macOS:

```bash
source .venv/bin/activate
```

Después de activar el entorno, ejecutar la instalación indicada anteriormente.

## Ejecución de pruebas

Desde `proyecto_radar/simulacion_python/`:

```bash
python -m pytest tests
```

Las pruebas verifican señales, ecos, DFT, FFT, IFFT, correlación, detección, espectro y estimación de distancia.

## Experimentos

Los experimentos deben ejecutarse desde `proyecto_radar/simulacion_python/`, porque las rutas de salida son relativas a ese directorio.

```bash
python experiments/experiment_timing.py
python experiments/experiment_effects.py
python experiments/experiment_echo_generation.py
python experiments/experiment_echo_effects.py
python experiments/experiment_correlation_timing.py
python experiments/experiment_ranging.py
```

Los resultados se guardan principalmente en `experiments/results/`, incluyendo figuras PNG y datos CSV.

## Módulos principales

### `src/radar/signals/`

- `signal.py`: define la estructura `Signal`, que contiene las muestras, la frecuencia de muestreo, el tiempo y la duración de una señal.
- `generators.py`: genera senoides, chirps lineales y ruido. El chirp es la señal acústica conocida que posteriormente se busca en la recepción.

### `src/radar/simulation/`

- `echoes.py`: simula el canal acústico. Crea copias retardadas y atenuadas de la señal transmitida, las superpone y puede agregar ruido gaussiano.

El modelo utilizado es:

$$
y[n] = \sum_i a_i x[n-d_i] + w[n],
$$

donde $d_i$ es el retardo, $a_i$ la atenuación y $w[n]$ el ruido.

### `src/radar/transforms/`

- `dft.py`: implementa la DFT directamente, con complejidad $O(N^2)$.
- `fft.py`: implementa una FFT recursiva radix-2 de Cooley-Tukey, con complejidad $O(N\log N)$ y longitudes potencia de dos.
- `ifft.py`: implementa la transformada inversa y permite regresar del dominio frecuencial al temporal.

### `src/radar/analysis/`

- `correlation.py`: calcula correlación cruzada directa y correlación mediante FFT. La versión frecuencial utiliza zero-padding para obtener una correlación lineal y evitar solapamiento circular.
- `detection.py`: encuentra el máximo de la magnitud de la correlación y convierte el lag detectado en muestras y segundos.
- `spectrum.py`: calcula eje de frecuencias, magnitud y fase del espectro, incluyendo la representación unilateral para señales reales.
- `timing.py`: mide tiempos de ejecución y resume promedio, mínimo, máximo y desviación estándar.
- `ranging.py`: convierte el retardo temporal en distancia usando la velocidad del sonido y la relación de ida y vuelta:

$$
d = \frac{c\tau}{2}.
$$

### `src/radar/utils/`

- `validation.py`: valida tamaños potencia de dos y calcula el siguiente tamaño apropiado para zero-padding.

### `src/radar/visualization/`

- `plots.py`: centraliza la generación y guardado de gráficas de espectros, señales, correlaciones, tiempos de ejecución, detección y ranging.

## Experimentos incluidos

- `experiment_timing.py`: compara los tiempos de la DFT y la FFT para tamaños de $N=16$ hasta $N=2048$ y genera un CSV y una gráfica.
- `experiment_effects.py`: muestra aliasing, spectral leakage y zero-padding.
- `experiment_echo_generation.py`: genera un chirp, varios ecos y recepciones con diferentes niveles de ruido.
- `experiment_echo_effects.py`: evalúa la detección de un eco conocido bajo ruido limpio, moderado y fuerte.
- `experiment_correlation_timing.py`: compara el tiempo de correlación directa frente a correlación mediante FFT.
- `experiment_ranging.py`: integra generación, ecos, ruido, correlación y detección para validar la estimación física de distancia, el efecto de la temperatura, la cuantización temporal, $P_D$ y $P_{FA}$.

## Resultados y documentación

- `proyecto_radar/simulacion_python/experiments/results/figures/`: figuras generadas por los experimentos.
- `proyecto_radar/simulacion_python/experiments/results/data/`: datos CSV de escenarios y mediciones.
- `proyecto_radar/simulacion_python/experiments/results/timing/`: resultados de benchmarks de DFT y FFT.
- [proyecto_radar/Hardware_ESP32/configuracion-hardware.MD](proyecto_radar/Hardware_ESP32/configuracion-hardware.MD): especificación técnica completa, esquemáticos y diseño del circuito de hardware.

## Montaje Experimental y Arquitectura de Hardware (ESP32-S3)

Además del modelado y la simulación numérica, el proyecto implementa un radar acústico monostático funcional en tiempo real sobre una placa **ESP32-S3 (Dual-Core LX7 a 240 MHz)**, validado experimentalmente sobre banco de pruebas.

### Resumen del Montaje Físico y Electrónico

* **Arquitectura de Procesamiento Concurrente (Dual-Core):**
  * **Core 0 (TX):** Síntesis de precisión de un pulso lineal modulado en frecuencia (*chirp* LFM de 100 pasos discretos, duración de $50\text{ ms}$, barrido de $2\text{ kHz}$ a $8\text{ kHz}$) modulado por PWM (`GPIO 4`).
  * **Core 1 (RX y DSP):** Adquisición analógica a $F_s = 24\text{ kHz}$ ($2400\text{ muestras}$ en $100\text{ ms}$, cumpliendo con creces Nyquist: $F_s/2 = 12\text{ kHz} > 8\text{ kHz}$) en `GPIO 1`, cálculo de correlación cruzada espectral mediante una **FFT Radix-2 de 4096 puntos**, extracción de envolvente analítica por Transformada de Hilbert y mitigación de ecos parásitos por **sustracción de perfil de fondo estático calibrado**.
* **Etapa de Potencia y Transmisión (TX):**
  * Elevación de potencia mediante un amplificador digital clase D **TPA3110** (alimentado a 12 V) acoplado a un altavoz/parlante de **8 $\Omega$**.
* **Etapa de Recepción y Acondicionamiento (RX):**
  * Captura de onda acústica mediante una cápsula de micrófono electret (**KY-038**).
  * **Preamplificador discreto de bajo ruido con transistor BJT BC549:** Configuración en emisor común con ganancia de voltaje $A_v \approx 40\text{ dB}$ (factor $\approx 100$), polarizado a 3.3 V y desacoplado para centrar el nivel DC en el rango óptimo del convertidor analógico-digital (ADC de 12 bits en `GPIO 1`).
* **Disposición Mecánica y Desacoplo Acústico:**
  * **Alineación Vertical:** El micrófono se encuentra ubicado en el mismo eje horizontal ($X$) y a una altura de $7\text{ a }8\text{ cm}$ directamente sobre el parlante ($Y$).
  * **Aislamiento Mecánico:** Montaje del micrófono sobre una cama de espuma de poliuretano de alta densidad para suprimir la propagación estructural de vibraciones desde la membrana del altavoz a través del chasis.
  * **Visera Anti-Acople Frontal:** Bafle de espuma de $2\text{ cm}$ de grosor por $6\text{ cm}$ de largo que sobresale hacia el frente entre el altavoz y el micrófono, atenuando drásticamente la onda directa (*crosstalk* acústico).
  * **Inclinación del Chasis:** Inclinación frontal hacia arriba de $\approx 0.8\text{ cm}$ para orientar el lóbulo acústico principal y evitar reflexiones rasantes parásitas sobre la mesa a distancias cortas ($30\text{ cm}$ a $45\text{ cm}$).
* **Blanco Reflector de Calibración:**
  * Disco plano de **36 cm de diámetro recubierto de papel de aluminio** (coeficiente de reflexión acústica $R \approx 1$), verificado a distancias reales de 30 cm, 35 cm, 40 cm y 45 cm.
* **Interfaz de Usuario (HMI):**
  * Pantalla LCD 16x2 vía bus I2C (**PCF8574** en `GPIO 8` SDA y `GPIO 9` SCL, dirección `0x3F`), desplegando en tiempo real la distancia estimada y el estado de adquisición.

### Explicación Técnica Detallada
Para consultar los esquemáticos circuitales del preamplificador en LTspice, valores exactos de polarización, componentes y detalles de calibración física, consulte la documentación técnica:  
👉 [proyecto_radar/Hardware_ESP32/configuracion-hardware.MD](proyecto_radar/Hardware_ESP32/configuracion-hardware.MD)

## Estructura del proyecto

```text
ffuchs_ymonge__asm_2026_s2/
├── README.md
└── proyecto_radar/
    ├── Hardware_ESP32/
    │   ├── Radar_Acoustic_Main/
    │   │   └── Radar_Acoustic_Main.ino       # Firmware principal del radar en ESP32-S3
    │   ├── configuracion-hardware.MD          # Especificación y montaje físico/electrónico
    │   ├── image.png                          # Diagrama esquemático del circuito preamplificador
    │   └── tests/                             # Pruebas unitarias de hardware
    │       ├── 01_tx_chirp_test/              # Síntesis y emisión del chirp TX
    │       ├── 02_rx_mic_adc_test/            # Muestreo y lectura ADC del micrófono
    │       ├── 03_loopback_tx_rx_test/        # Prueba de loopback continuo TX -> RX
    │       ├── 04_matched_filter_radar_test/  # Prototipo inicial de filtro adaptado
    │       └── 05_i2c_lcd_test.ino/           # Verificación de pantalla LCD 16x2 I2C
    └── simulacion_python/
        ├── pyproject.toml                     # Configuración del paquete Python
        ├── requirements.txt                   # Dependencias del proyecto
        ├── experiments/                       # Scripts de experimentación y benchmarks
        │   ├── experiment_timing.py           # Comparación computacional DFT vs FFT
        │   ├── experiment_effects.py          # Aliasing, fuga espectral y zero-padding
        │   ├── experiment_echo_generation.py  # Síntesis de escenarios de ecos acústicos
        │   ├── experiment_echo_effects.py     # Detección bajo diversos niveles de SNR
        │   ├── experiment_correlation_timing.py # Benchmarks de correlación temporal vs FFT
        │   ├── experiment_ranging.py          # Estimación de distancia, ToF y métricas
        │   └── results/                       # Figuras generadas (PNG) y datos (CSV)
        ├── src/radar/                         # Paquete modular DSP en Python
        │   ├── analysis/                      # Correlación, picos, espectro y ranging ToF
        │   ├── signals/                       # Generadores de señal y chirp LFM
        │   ├── simulation/                    # Modelado del canal acústico y ecos
        │   ├── transforms/                    # Implementación de DFT directa y FFT/IFFT radix-2
        │   ├── utils/                         # Validación y zero-padding
        │   └── visualization/                 # Gráficas de resultados
        └── tests/                             # Pruebas unitarias automatizadas (pytest)
```
