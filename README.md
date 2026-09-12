# Sistema de radar acústico

Proyecto de Procesamiento Digital de Señales para estimar distancias mediante una señal acústica conocida, simulación de ecos, correlación y análisis espectral.

El proyecto está implementado en Python y compara una DFT directa con una FFT radix-2 propia. También estudia aliasing, fuga espectral, zero-padding, ruido, detección de ecos y estimación de distancia.

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
- `proyecto_radar/simulacion_python/DOCUMENTACION_ENTREGABLES.md`: explicación matemática y técnica organizada según los entregables del proyecto.

## Estructura del proyecto

```
ffuchs_ymonge__asm_2026_s2
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        └─ nodeids
├─ proyecto_radar
│  └─ simulacion_python
│     ├─ .pytest_cache
│     │  ├─ CACHEDIR.TAG
│     │  ├─ README.md
│     │  └─ v
│     │     └─ cache
│     │        ├─ lastfailed
│     │        └─ nodeids
│     ├─ DOCUMENTACION_ENTREGABLES.md
│     ├─ experiments
│     │  ├─ experiment_correlation_timing.py
│     │  ├─ experiment_echo_effects.py
│     │  ├─ experiment_echo_generation.py
│     │  ├─ experiment_effects.py
│     │  ├─ experiment_ranging.py
│     │  ├─ experiment_timing.py
│     │  └─ results
│     │     ├─ data
│     │     │  └─ echo_scenario.csv
│     │     ├─ figures
│     │     │  ├─ aliasing
│     │     │  │  └─ espectro_aliasing_700hz.png
│     │     │  ├─ correlation
│     │     │  │  ├─ deteccion_pico_limpio.png
│     │     │  │  ├─ deteccion_pico_ruido_fuerte.png
│     │     │  │  └─ deteccion_pico_ruido_moderado.png
│     │     │  ├─ echoes
│     │     │  │  ├─ recibida_limpia.png
│     │     │  │  ├─ recibida_ruido_fuerte.png
│     │     │  │  ├─ recibida_ruido_moderado.png
│     │     │  │  └─ tx_chirp_zoom_5ms.png
│     │     │  ├─ leakage
│     │     │  │  ├─ leakage_01_perfect_50hz.png
│     │     │  │  └─ leakage_02_leaking_50_5hz.png
│     │     │  ├─ ranging
│     │     │  │  ├─ distance_validation.png
│     │     │  │  ├─ false_alarm_threshold.png
│     │     │  │  ├─ pd_vs_snr.png
│     │     │  │  ├─ quantization_error.png
│     │     │  │  └─ thermal_compensation.png
│     │     │  ├─ timing
│     │     │  │  ├─ benchmark_correlacion_comparativo.png
│     │     │  │  └─ benchmark_timing_comparativo.png
│     │     │  └─ zero_padding
│     │     │     ├─ zp_01_corta_N64.png
│     │     │     └─ zp_02_rellena_N1024.png
│     │     └─ timing
│     │        └─ dft_fft_benchmark.csv
│     ├─ pyproject.toml
│     ├─ requirements.txt
│     ├─ results
│     │  ├─ data
│     │  ├─ figures
│     │  └─ timing
│     ├─ src
│     │  ├─ radar
│     │  │  ├─ analysis
│     │  │  │  ├─ correlation.py
│     │  │  │  ├─ detection.py
│     │  │  │  ├─ ranging.py
│     │  │  │  ├─ spectrum.py
│     │  │  │  ├─ timing.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ correlation.cpython-313.pyc
│     │  │  │     ├─ detection.cpython-313.pyc
│     │  │  │     ├─ ranging.cpython-313.pyc
│     │  │  │     ├─ spectrum.cpython-313.pyc
│     │  │  │     ├─ timing.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ signals
│     │  │  │  ├─ generators.py
│     │  │  │  ├─ signal.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ generators.cpython-313.pyc
│     │  │  │     ├─ signal.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ simulation
│     │  │  │  ├─ echoes.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ echoes.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ transforms
│     │  │  │  ├─ dft.py
│     │  │  │  ├─ fft.py
│     │  │  │  ├─ ifft.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ dft.cpython-313.pyc
│     │  │  │     ├─ fft.cpython-313.pyc
│     │  │  │     ├─ ifft.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ utils
│     │  │  │  ├─ validation.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ validation.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ visualization
│     │  │  │  ├─ plots.py
│     │  │  │  ├─ __init__.py
│     │  │  │  └─ __pycache__
│     │  │  │     ├─ plots.cpython-313.pyc
│     │  │  │     └─ __init__.cpython-313.pyc
│     │  │  ├─ __init__.py
│     │  │  └─ __pycache__
│     │  │     └─ __init__.cpython-313.pyc
│     │  └─ radar.egg-info
│     │     ├─ dependency_links.txt
│     │     ├─ PKG-INFO
│     │     ├─ requires.txt
│     │     ├─ SOURCES.txt
│     │     └─ top_level.txt
│     └─ tests
│        ├─ test_correlation.py
│        ├─ test_detection.py
│        ├─ test_echoes.py
│        ├─ test_ranging.py
│        ├─ test_signal.py
│        ├─ test_spectrum.py
│        ├─ test_transforms.py
│        └─ __pycache__
│           ├─ test_correlation.cpython-313-pytest-9.1.1.pyc
│           ├─ test_detection.cpython-313-pytest-9.1.1.pyc
│           ├─ test_echoes.cpython-313-pytest-9.1.1.pyc
│           ├─ test_ranging.cpython-313-pytest-9.1.1.pyc
│           ├─ test_signal.cpython-313-pytest-9.1.1.pyc
│           ├─ test_spectrum.cpython-313-pytest-9.1.1.pyc
│           └─ test_transforms.cpython-313-pytest-9.1.1.pyc
└─ README.md

```
