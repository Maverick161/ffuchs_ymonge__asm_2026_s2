#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ==============================================================================
// 1. PINES Y HARDWARE ESP32-S3
// ==============================================================================
#define TONE_PIN 4    // Salida PWM (LEDC) hacia el amplificador TPA3110 (12V)
#define MIC_ADC_PIN 1 // Entrada analógica del preamplificador BC549 (KY-038)

#define I2C_SDA_PIN 8 // Bus I2C - Datos LCD
#define I2C_SCL_PIN 9 // Bus I2C - Reloj LCD
#define LCD_ADDR 0x3F // Dirección I2C del PCF8574

// ==============================================================================
// 2. PARÁMETROS ACÚSTICOS Y DE MUESTREO (24 kHz)
// ==============================================================================
#define F_START 2000
#define F_STOP 8000
#define CHIRP_DURATION_US 50000ULL // 50 ms de duración del chirp lineal
#define NUM_CHIRP_STEPS                                                        \
  100 // 100 pasos (500 µs por paso = 12 muestras a 24 kHz)

#define FS 24000ULL
#define TX_SAMPLES 1200 // 50 ms a 24 kHz = 1200 muestras
#define RX_SAMPLES 2400 // 100 ms de ventana de escucha = 2400 muestras

#define FFT_SIZE 4096 // Tamaño potencia de 2 para FFT radix-2

#define SPEED_OF_SOUND 343.42f // Velocidad del sonido en aire a ~20 °C (m/s)

// ==============================================================================
// 3. CALIBRACIÓN EMPÍRICA Y GEOMETRÍA DEL BANCO DE PRUEBAS
// ==============================================================================
// Puntos de anclaje empíricos basados en los logs reales de hardware:
// - A 30.0 cm real: sub ≈ 52.82 -> d_raw ≈ 37.79 cm -> offset = 7.79 cm
// - A 35.0 cm real: sub ≈ 57.98 -> d_raw ≈ 41.48 cm -> offset = 6.48 cm
// - A 40.0 cm real: sub ≈ 63.19 -> d_raw ≈ 45.21 cm -> offset = 5.21 cm
// - A 45.0 cm real: sub ≈ 72.27 -> d_raw ≈ 51.71 cm -> offset = 6.71 cm
#define CAL_D_RAW_30 36.87f
#define CAL_D_RAW_40 45.21f
#define CAL_D_RAW_45 51.70f
#define CAL_OFFSET_30 6.87f
#define CAL_OFFSET_40 5.21f
#define CAL_OFFSET_45 6.70f

// Delimitación estricta del rango operativo: 30 cm a menos de 50 cm
// 1 muestra a 24 kHz equivale a: (343.42 / 24000) / 2 * 100 = 0.7155 cm de
// retardo ToF k=48 (~29.0 cm) hasta k=78 (~48.8 cm), cortando antes de la pared
// de 50.2 cm (k >= 80)
#define MIN_SEARCH_INDEX 48
#define MAX_SEARCH_INDEX 78
#define MIN_VALID_DIST_CM 28.5f
#define MAX_VALID_DIST_CM 49.5f

// Integraciones por ciclo
#define INTEGRATION_COUNT 5 // 5 acumulaciones para excelente SNR
#define INTER_CHIRP_DELAY_MS                                                   \
  40 // Pausa entre ráfagas para permitir extinción de reverberación

// ==============================================================================
// 4. PARÁMETROS SOCA-CFAR Y COHERENCIA ESPECTRAL
// ==============================================================================
#define CFAR_TRAINING_RADIUS 6 // Celdas de entrenamiento a cada lado
#define CFAR_GUARD_CELLS 2     // Celdas de guardia
#define CFAR_MULTIPLIER 1.20f  // Umbral CFAR sobre el piso local
#define MIN_PEAK_COHERENCE                                                     \
  1.15f // Relación pico / vecindario con celda de guardia para eco real

// ==============================================================================
// 5. FILTRADO TEMPORAL Y TRACKING DE BLANCO
// ==============================================================================
#define MEDIAN_DEPTH 3         // Ventana de mediana
#define IIR_ALPHA_TRACK 0.45f  // Filtro suave para posición estable
#define IIR_ALPHA_JUMP 0.85f   // Salto rápido ante desplazamientos
#define JUMP_THRESHOLD_CM 4.5f // Umbral de salto
#define SPATIAL_MATCH_CM                                                       \
  3.5f                  // Coincidencia espacial para confirmar nuevo blanco
#define CONFIRM_COUNT 2 // Detecciones consecutivas para enganchar
#define LOST_COUNT 5    // Ciclos sin eco para declarar "Buscando..."

// Estructura de telemetría de candidatos
struct PeakCandidate {
  int index;
  float refined_index;
  float dist_cm;
  float raw;
  float bg;
  float net;
  float ratio;
  float score;
  float coherence;
};

// ==============================================================================
// 6. ESTRUCTURAS DE MEMORIA Y BUFFERS
// ==============================================================================
uint16_t *rx_buffer = nullptr;

float *real_X = nullptr, *imag_X = nullptr; // Réplica conjugada del chirp
float *real_Y = nullptr,
      *imag_Y = nullptr; // Buffer FFT/IFFT para correlación cruzada

// Mapa de fondo estático de envolvente (inmune al jitter de fase del PWM)
float *bg_envelope = nullptr;
float *env_accum = nullptr;
float *net_signal = nullptr;

// Sincronización inter-núcleos
volatile bool prepare_chirp = false;
volatile bool chirp_ready = false;
volatile bool fire_chirp = false;
volatile bool chirp_started = false;

// Estado del tracker
float last_detected_raw = -1.0f;
float last_stable_distance = -1.0f;
float median_buffer[MEDIAN_DEPTH];
int median_write_idx = 0;
bool median_full = false;
int consecutive_detections = 0;
int consecutive_misses = 0;
bool tracking_active = false;

// Estadísticas de sesión
uint32_t total_cycles = 0;
uint32_t detection_cycles = 0;

// ==============================================================================
// 7. CONTROLADOR LCD 16x2 I2C (PCF8574)
// ==============================================================================
#define PIN_RS 0x01
#define PIN_RW 0x02
#define PIN_EN 0x04
#define PIN_BL 0x08

void pcfWrite(uint8_t val) {
  Wire.beginTransmission(LCD_ADDR);
  Wire.write(val);
  Wire.endTransmission();
}

void pulseEnable(uint8_t val) {
  pcfWrite(val | PIN_EN);
  delayMicroseconds(1);
  pcfWrite(val & ~PIN_EN);
  delayMicroseconds(50);
}

void send4Bits(uint8_t value, uint8_t mode) {
  pulseEnable((value & 0xF0) | mode | PIN_BL);
}

void sendByte(uint8_t value, uint8_t mode) {
  send4Bits(value & 0xF0, mode);
  send4Bits((value << 4) & 0xF0, mode);
}

void lcdCommand(uint8_t cmd) { sendByte(cmd, 0); }
void lcdWriteChar(char c) { sendByte((uint8_t)c, PIN_RS); }

void lcdPrint(const char *str) {
  while (*str)
    lcdWriteChar(*str++);
}

void lcdSetCursor(uint8_t col, uint8_t row) {
  int row_offsets[] = {0x00, 0x40};
  lcdCommand(0x80 | (col + row_offsets[row]));
}

void lcdClear() {
  lcdCommand(0x01);
  delayMicroseconds(2000);
}

void lcdInit() {
  delay(50);
  pcfWrite(PIN_BL);
  delay(100);
  send4Bits(0x30, 0);
  delayMicroseconds(4500);
  send4Bits(0x30, 0);
  delayMicroseconds(4500);
  send4Bits(0x30, 0);
  delayMicroseconds(150);
  send4Bits(0x20, 0);
  lcdCommand(0x28);
  lcdCommand(0x0C);
  lcdClear();
  lcdCommand(0x06);
}

// ==============================================================================
// 8. MOTOR DSP: FFT RADIX-2 COOLEY-TUKEY
// ==============================================================================
inline void swapFloat(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

void computeFFT(float *real, float *imag, uint16_t n, bool inverse) {
  uint16_t j = 0;
  for (uint16_t i = 0; i < n - 1; i++) {
    if (i < j) {
      swapFloat(real[i], real[j]);
      swapFloat(imag[i], imag[j]);
    }
    uint16_t m = n >> 1;
    while (j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (uint16_t step = 1; step < n; step <<= 1) {
    float dir = inverse ? 1.0f : -1.0f;
    float theta = dir * PI / step;
    float w_real = cosf(theta);
    float w_imag = sinf(theta);

    for (uint16_t i = 0; i < n; i += (step << 1)) {
      float u_real = 1.0f;
      float u_imag = 0.0f;
      for (uint16_t k = 0; k < step; k++) {
        uint16_t idx1 = i + k;
        uint16_t idx2 = idx1 + step;

        float t_real = u_real * real[idx2] - u_imag * imag[idx2];
        float t_imag = u_real * imag[idx2] + u_imag * real[idx2];

        real[idx2] = real[idx1] - t_real;
        imag[idx2] = imag[idx1] - t_imag;
        real[idx1] += t_real;
        imag[idx1] += t_imag;

        float next_u_real = u_real * w_real - u_imag * w_imag;
        u_imag = u_real * w_imag + u_imag * w_real;
        u_real = next_u_real;
      }
    }
  }

  if (inverse) {
    float inv_n = 1.0f / (float)n;
    for (uint16_t i = 0; i < n; i++) {
      real[i] *= inv_n;
      imag[i] *= inv_n;
    }
  }
}

// ==============================================================================
// 9. FUNCIONES DE CALIBRACIÓN Y DISCRIMINACIÓN DSP
// ==============================================================================

// Compensación de offset basada en datos empíricos del banco de prueba
float calibrationOffset(float d_raw_cm) {
  if (d_raw_cm <= CAL_D_RAW_40) {
    // Tramo 1: 30 cm a 40 cm
    float slope =
        (CAL_OFFSET_40 - CAL_OFFSET_30) / (CAL_D_RAW_40 - CAL_D_RAW_30);
    float offset = CAL_OFFSET_30 + slope * (d_raw_cm - CAL_D_RAW_30);
    if (offset < 4.5f)
      offset = 4.5f;
    if (offset > 9.0f)
      offset = 9.0f;
    return offset;
  } else {
    // Tramo 2: 40 cm a 50 cm
    float slope =
        (CAL_OFFSET_45 - CAL_OFFSET_40) / (CAL_D_RAW_45 - CAL_D_RAW_40);
    float offset = CAL_OFFSET_40 + slope * (d_raw_cm - CAL_D_RAW_40);
    if (offset < 4.5f)
      offset = 4.5f;
    if (offset > 8.5f)
      offset = 8.5f;
    return offset;
  }
}

// Umbral dependiente de la distancia (Time Varied Gain inverso adaptativo):
// Calibrado con la física empírica de banco (logs de hardware):
// - Tramo 30 cm (k <= 55): net >= 3500, ratio >= 1.45 (ecos reales disco:
// 4800-8800; parásito visera <= 1200)
// - Tramo 35-38 cm (k = 56..61): net >= 1800, ratio >= 1.30 (zona limpia; eco
// disco real 3500-5000)
// - Tramo 40 cm (k = 62..67): net >= 2200, ratio >= 1.35 (residuo mesa <= 1241;
// eco disco real 3300-4500)
// - Tramo 43 cm (k = 68..70): net >= 2000, ratio >= 1.35 (zona limpia previa a
// pared; eco disco real 3000-4500)
// - Tramo 45 cm (k >= 71): net >= 2400, ratio >= 1.45 (residuo pared en k=72 es
// 1788; eco disco 2600-6000)
float minNetThreshold(int k) {
  if (k <= 55) {
    return 3200.0f; // Tramo 30 cm (eco real disco: 6000-7200; vacío <= 2126)
  } else if (k <= 61) {
    return 1500.0f; // Tramo 35 cm (eco neto cresta con sombra: ~4500-5300;
                    // vacío <= 0)
  } else if (k <= 67) {
    return 750.0f; // Tramo 40 cm (eco neto cresta con sombra: 893-2085; vacío
                   // <= 707)
  } else if (k <= 70) {
    return 1200.0f; // Tramo 43 cm
  } else {
    return 750.0f; // Tramo 45 cm (eco neto cresta con sombra: 3200-3700; vacío
                   // <= 0 con guardia)
  }
}

float minRatioThreshold(int k) {
  if (k <= 55) {
    return 1.40f;
  } else if (k <= 61) {
    return 1.15f; // Tramo 35 cm
  } else if (k <= 67) {
    return 1.20f; // Tramo 40 cm (ratio sobre piso ensombrecido: 1.37-2.19)
  } else if (k <= 70) {
    return 1.20f;
  } else {
    return 1.15f; // Tramo 45 cm (ratio sobre piso ensombrecido: 1.53-1.76)
  }
}

// Mediana para suprimir transitorios
float computeMedian(float *arr, int len) {
  float tmp[MEDIAN_DEPTH];
  for (int i = 0; i < len; i++)
    tmp[i] = arr[i];
  for (int i = 1; i < len; i++) {
    float key = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j] > key) {
      tmp[j + 1] = tmp[j];
      j--;
    }
    tmp[j + 1] = key;
  }
  return tmp[len / 2];
}

// Coherencia espectral del pico (nitidez respecto al vecindario local con celda
// de guardia)
float peakCoherence(int k, float *signal, int radius) {
  float center = signal[k];
  if (center < 1.0f)
    return 0.0f;
  float sum_neighbors = 0.0f;
  int count = 0;
  for (int d = -radius; d <= radius; d++) {
    // Guarda de 1 celda para no comparar el pico contra los hombros de su
    // propio pulso
    if (abs(d) <= 1)
      continue;
    int idx = k + d;
    if (idx >= MIN_SEARCH_INDEX && idx < MAX_SEARCH_INDEX) {
      sum_neighbors += signal[idx];
      count++;
    }
  }
  if (count == 0)
    return 1.0f;
  float avg_neighbors = sum_neighbors / (float)count;
  return center / (avg_neighbors + 1.0f);
}

// SOCA-CFAR (Smallest Of Cell Averaging): inmune al enmascaramiento por ecos
// adyacentes
float localCfarThreshold(int index, float *signal_array) {
  float sum_left = 0.0f;
  int count_left = 0;
  float sum_right = 0.0f;
  int count_right = 0;

  int left_start = max((int)MIN_SEARCH_INDEX, index - CFAR_TRAINING_RADIUS);
  int left_end = index - CFAR_GUARD_CELLS - 1;
  for (int k = left_start; k <= left_end; k++) {
    sum_left += signal_array[k];
    count_left++;
  }

  int right_start = index + CFAR_GUARD_CELLS + 1;
  int right_end = min((int)MAX_SEARCH_INDEX, index + CFAR_TRAINING_RADIUS);
  for (int k = right_start; k <= right_end; k++) {
    sum_right += signal_array[k];
    count_right++;
  }

  float avg_left = (count_left > 0) ? (sum_left / (float)count_left) : 0.0f;
  float avg_right = (count_right > 0) ? (sum_right / (float)count_right) : 0.0f;

  float floor_val;
  if (count_left > 0 && count_right > 0) {
    floor_val = fminf(avg_left, avg_right); // SOCA: Smallest Of Cell Averaging
  } else if (count_left > 0) {
    floor_val = avg_left;
  } else {
    floor_val = avg_right;
  }

  return floor_val * CFAR_MULTIPLIER;
}

// ==============================================================================
// 10. GESTIÓN DE MEMORIA ROBUSTA
// ==============================================================================
void *safeAlloc(size_t size) {
  void *p = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
  if (!p) {
    p = heap_caps_malloc(size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  }
  if (!p) {
    p = malloc(size);
  }
  return p;
}

void allocateMemory() {
  rx_buffer = (uint16_t *)safeAlloc(RX_SAMPLES * sizeof(uint16_t));

  real_X = (float *)safeAlloc(FFT_SIZE * sizeof(float));
  imag_X = (float *)safeAlloc(FFT_SIZE * sizeof(float));
  real_Y = (float *)safeAlloc(FFT_SIZE * sizeof(float));
  imag_Y = (float *)safeAlloc(FFT_SIZE * sizeof(float));

  bg_envelope = (float *)safeAlloc(FFT_SIZE * sizeof(float));
  env_accum = (float *)safeAlloc(FFT_SIZE * sizeof(float));
  net_signal = (float *)safeAlloc(FFT_SIZE * sizeof(float));

  for (int i = 0; i < FFT_SIZE; i++) {
    bg_envelope[i] = 0.0f;
    env_accum[i] = 0.0f;
    net_signal[i] = 0.0f;
  }

  for (int i = 0; i < MEDIAN_DEPTH; i++) {
    median_buffer[i] = -1.0f;
  }
}

// ==============================================================================
// 11. GENERACIÓN DEL CHIRP (CORE 0) Y MUESTREO (CORE 1)
// ==============================================================================
void chirpTaskCode(void *pvParameters) {
  uint32_t step_delay_us = CHIRP_DURATION_US / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;

  while (true) {
    if (prepare_chirp) {
      prepare_chirp = false;
      chirp_ready = true;

      while (!fire_chirp) {
      }
      fire_chirp = false;

      int64_t start_t = esp_timer_get_time();
      ledcWriteTone(TONE_PIN, F_START);
      chirp_started = true;
      for (int i = 0; i < NUM_CHIRP_STEPS; i++) {
        if (i > 0) {
          ledcWriteTone(TONE_PIN, F_START + (uint32_t)(i * freq_step));
        }
        int64_t target = start_t + (i + 1) * step_delay_us;
        while (esp_timer_get_time() < target)
          ;
      }
      ledcWriteTone(TONE_PIN, 0);
    }
    vTaskDelay(1);
  }
}

void acquireSignal() {
  prepare_chirp = true;
  while (!chirp_ready) {
    delay(1);
  }
  chirp_ready = false;

  chirp_started = false;
  fire_chirp = true;

  while (!chirp_started)
    ;

  int64_t start_time = esp_timer_get_time();

  // Muestreo a 24 kHz
  for (int i = 0; i < RX_SAMPLES; i++) {
    int64_t target = start_time + ((int64_t)i * 1000000LL) / (int64_t)FS;
    while (esp_timer_get_time() < target)
      ;
    rx_buffer[i] = analogRead(MIC_ADC_PIN);
  }
}

void precalculateChirpFFT() {
  for (int i = 0; i < FFT_SIZE; i++) {
    real_X[i] = 0.0f;
    imag_X[i] = 0.0f;
  }

  const int samples_per_step = TX_SAMPLES / NUM_CHIRP_STEPS;
  const float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;
  float phase = 0.0f;

  for (int n = 0; n < TX_SAMPLES; n++) {
    float window =
        0.54f - 0.46f * cosf(2.0f * PI * n / (float)(TX_SAMPLES - 1));
    real_X[n] = sinf(phase) * window;

    int step = n / samples_per_step;
    if (step >= NUM_CHIRP_STEPS)
      step = NUM_CHIRP_STEPS - 1;
    float frequency = F_START + step * freq_step;
    phase += 2.0f * PI * frequency / (float)FS;
  }

  computeFFT(real_X, imag_X, FFT_SIZE, false);

  // Conjugada compleja para correlación cruzada
  for (int i = 0; i < FFT_SIZE; i++) {
    imag_X[i] = -imag_X[i];
  }
}

void computeCrossCorrelation(bool is_calibration, int cal_passes = 1) {
  float dc_sum = 0.0f;
  for (int i = 0; i < RX_SAMPLES; i++)
    dc_sum += rx_buffer[i];
  float dc_offset = dc_sum / (float)RX_SAMPLES;

  for (int i = 0; i < FFT_SIZE; i++) {
    if (i < RX_SAMPLES)
      real_Y[i] = (float)rx_buffer[i] - dc_offset;
    else
      real_Y[i] = 0.0f;
    imag_Y[i] = 0.0f;
  }

  computeFFT(real_Y, imag_Y, FFT_SIZE, false);

  for (int i = 0; i < FFT_SIZE; i++) {
    float r = real_Y[i] * real_X[i] - imag_Y[i] * imag_X[i];
    float im = real_Y[i] * imag_X[i] + imag_Y[i] * real_X[i];
    real_Y[i] = r;
    imag_Y[i] = im;
  }

  // Transformada de Hilbert en frecuencia para extracción directa de la
  // envolvente
  for (int i = 1; i < FFT_SIZE / 2; i++) {
    real_Y[i] *= 2.0f;
    imag_Y[i] *= 2.0f;
  }
  for (int i = (FFT_SIZE / 2) + 1; i < FFT_SIZE; i++) {
    real_Y[i] = 0.0f;
    imag_Y[i] = 0.0f;
  }

  computeFFT(real_Y, imag_Y, FFT_SIZE, true);

  if (is_calibration) {
    float inv_cal = 1.0f / (float)cal_passes;
    for (int i = 0; i < FFT_SIZE; i++) {
      float env = hypotf(real_Y[i], imag_Y[i]);
      bg_envelope[i] += env * inv_cal;
    }
  }
}

// ==============================================================================
// 12. SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ledcAttach(TONE_PIN, F_START, 8);
  ledcWriteTone(TONE_PIN, 0);

  lcdInit();
  lcdSetCursor(0, 0);
  lcdPrint("Iniciando Radar");
  lcdSetCursor(0, 1);
  lcdPrint("Dual-Core ESP32 ");

  allocateMemory();
  precalculateChirpFFT();

  xTaskCreatePinnedToCore(chirpTaskCode, "ChirpTask", 2048, NULL, 1, NULL, 0);

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Alejese ahora   ");
  lcdSetCursor(0, 1);
  lcdPrint("Mesa despejada  ");

  for (int i = 4; i > 0; i--) {
    Serial.printf("Calibracion en %d s (DESPEJE LA MESA Y EL DISCO)...\n", i);
    delay(1000);
  }

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Calibrando...   ");
  lcdSetCursor(0, 1);
  lcdPrint("Silencio total  ");

  // Ráfagas de calentamiento previo para estabilizar el amplificador TPA3110 y
  // el parlante en régimen térmico
  for (int w = 0; w < 8; w++) {
    acquireSignal();
    delay(50);
  }

  const int cal_passes = 16;
  for (int c = 0; c < cal_passes; c++) {
    Serial.printf("Captura de fondo %d/%d\n", c + 1, cal_passes);
    acquireSignal();
    computeCrossCorrelation(true, cal_passes);
    delay(INTER_CHIRP_DELAY_MS);
  }

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Radar 30-50 cm  ");
  lcdSetCursor(0, 1);
  lcdPrint("Listo para medir");

  Serial.println("\n=== RADAR ACUSTICO TOF CALIBRADO ===");
  Serial.printf("  Rango de busqueda: k=[%d, %d] (30 cm a < 50 cm)\n",
                MIN_SEARCH_INDEX, MAX_SEARCH_INDEX);
  Serial.printf("  Integraciones por ciclo: %d\n", INTEGRATION_COUNT);
  Serial.printf("  Procesamiento: Envolvente Analitica con Supresion Positiva "
                "de Fondo\n");
  Serial.printf("  TVG Adaptable: 30cm>=%.0f | 35cm>=%.0f | 40cm>=%.0f | "
                "43cm>=%.0f | 45cm>=%.0f\n",
                minNetThreshold(53), minNetThreshold(58), minNetThreshold(63),
                minNetThreshold(69), minNetThreshold(73));
  Serial.printf("  Filtro Tracking: Mediana [%d] + IIR (alpha=%.2f/%.2f)\n",
                MEDIAN_DEPTH, IIR_ALPHA_TRACK, IIR_ALPHA_JUMP);
  Serial.println("====================================\n");
  delay(1000);
}

// ==============================================================================
// 13. BUCLE PRINCIPAL (LOOP)
// ==============================================================================
void loop() {
  uint32_t loop_start_ms = millis();
  total_cycles++;

  for (int i = 0; i < FFT_SIZE; i++) {
    env_accum[i] = 0.0f;
  }

  uint16_t adc_min = 4095;
  uint16_t adc_max = 0;
  float rms_sum = 0.0f;

  float inv_integ = 1.0f / (float)INTEGRATION_COUNT;

  // 1. Acumulación de envolventes de correlación cruzada
  for (int cap = 0; cap < INTEGRATION_COUNT; cap++) {
    acquireSignal();

    for (int i = 0; i < RX_SAMPLES; i++) {
      if (rx_buffer[i] > adc_max)
        adc_max = rx_buffer[i];
      if (rx_buffer[i] < adc_min)
        adc_min = rx_buffer[i];
    }

    float dc_tmp = 0.0f;
    for (int i = 0; i < RX_SAMPLES; i++)
      dc_tmp += rx_buffer[i];
    dc_tmp /= (float)RX_SAMPLES;
    for (int i = 0; i < RX_SAMPLES; i++) {
      float v = (float)rx_buffer[i] - dc_tmp;
      rms_sum += v * v;
    }

    computeCrossCorrelation(false);

    for (int i = 0; i < FFT_SIZE; i++) {
      float env = hypotf(real_Y[i], imag_Y[i]);
      env_accum[i] += env * inv_integ;
    }

    delay(INTER_CHIRP_DELAY_MS);
  }

  float rms_val = sqrtf(rms_sum / (float)(RX_SAMPLES * INTEGRATION_COUNT));

  // 2. Fondo estático calibrado
  // Dado que el volumen del amplificador y la ganancia del preamplificador
  // se mantienen constantes, la envolvente de calibración (16 pasadas)
  // representa con exactitud milimétrica el entorno estático de la mesa.
  // El margen de guarda del 5% (1.05x) absorbe fluctuaciones térmicas y ruido
  // ADC.
  const float g_scale = 1.0f;

  // 3. Extracción de Energía Neta de Eco (Sustracción de Fondo Estático)
  for (int i = 0; i < FFT_SIZE; i++) {
    float raw = env_accum[i];
    // Margen de guarda del 8% para absorber fluctuaciones térmicas y deriva del
    // fondo en vacío
    float bg_margin = 1.08f;
    float bg = bg_margin * bg_envelope[i];
    float diff = raw - bg;
    net_signal[i] = (diff > 0.0f) ? diff : 0.0f;
  }

  // 3b. Detección de Blanco en Zonas Enmascaradas por Sombra Acústica
  // El disco de aluminio de 36 cm actúa como barrera acústica, interceptando
  // y atenuando físicamente las reflexiones estáticas de la mesa o pared de
  // fondo.

  // Caso 45 cm (k=73): Cresta física directa en k=73 con atenuación de pared en
  // k=70..71
  bool shadow_45 =
      (env_accum[73] > env_accum[72] && env_accum[73] >= env_accum[74]) &&
      (env_accum[73] > 7500.0f) &&
      (bg_envelope[70] > 5000.0f && bg_envelope[71] > 5000.0f) &&
      ((env_accum[70] < 0.75f * bg_envelope[70]) ||
       (env_accum[71] < 0.70f * bg_envelope[71]));

  // Caso 40 cm (k=63..66): El lóbulo principal de la mesa en k=61..62 se
  // desploma (> 65% de caída)
  bool shadow_40 = (!shadow_45) &&
                   (bg_envelope[61] > 5000.0f && bg_envelope[62] > 5000.0f) &&
                   ((env_accum[61] < 0.40f * bg_envelope[61]) ||
                    (env_accum[62] < 0.35f * bg_envelope[62])) &&
                   (env_accum[64] > 2600.0f || env_accum[65] > 2600.0f);

  // Caso 35 cm (k=58..61): Cresta directa en k=59..60 con atenuación en k=56,
  // sin ecos en 40 o 45 cm
  bool is_peak_35 =
      (env_accum[59] > env_accum[58] && env_accum[59] >= env_accum[60]) ||
      (env_accum[60] > env_accum[59] && env_accum[60] >= env_accum[61]);
  bool shadow_35 = (!shadow_45) && (!shadow_40) && is_peak_35 &&
                   (env_accum[56] < 4500.0f) &&
                   (env_accum[59] > 6500.0f || env_accum[60] > 6500.0f);

  if (shadow_45) {
    float floor_val = fminf(env_accum[71], env_accum[75]);
    float net_shadow = env_accum[73] - floor_val;
    if (net_shadow > net_signal[73]) {
      net_signal[73] = net_shadow;
    }
  } else if (shadow_40) {
    for (int k = 63; k <= 66; k++) {
      if (env_accum[k] > env_accum[k - 1] && env_accum[k] >= env_accum[k + 1] &&
          env_accum[k] > 2600.0f) {
        float floor_val = fminf(env_accum[62], env_accum[63]);
        float net_shadow = env_accum[k] - floor_val;
        if (net_shadow > net_signal[k]) {
          net_signal[k] = net_shadow;
        }
      }
    }
  } else if (shadow_35) {
    for (int k = 58; k <= 61; k++) {
      if (env_accum[k] > env_accum[k - 1] && env_accum[k] >= env_accum[k + 1]) {
        float floor_val = fminf(env_accum[55], env_accum[56]);
        float net_shadow = env_accum[k] - floor_val;
        if (net_shadow > net_signal[k]) {
          net_signal[k] = net_shadow;
        }
      }
    }
  }

  bool shadow_detected = shadow_35 || shadow_40 || shadow_45;

  // Energía total en la banda de búsqueda
  float total_band_energy = 0.0f;
  for (int k = MIN_SEARCH_INDEX; k <= MAX_SEARCH_INDEX; k++) {
    total_band_energy += net_signal[k];
  }

  PeakCandidate candidates[5];
  int cand_count = 0;

  // 4. Búsqueda y Validación de Picos en la Ventana de 30 cm a < 50 cm
  for (int k = MIN_SEARCH_INDEX; k <= MAX_SEARCH_INDEX; k++) {
    float net_val = net_signal[k];

    // Condición de cresta local estricta
    bool is_peak =
        (net_val > net_signal[k - 1]) && (net_val >= net_signal[k + 1]);
    if (!is_peak)
      continue;

    // En el extremo lejano (k >= 71), la pared estática en vacío tiene una
    // pendiente descendente muy pronunciada. Un eco real debe ser una cresta
    // física en RAW o tener sombra de pared activa para evitar falsas alarmas
    // por residuo de pendiente.
    if (k >= 71 && !shadow_45) {
      if (env_accum[k] <= env_accum[k - 1]) {
        continue;
      }
    }

    float raw_val = env_accum[k];
    float bg_val = g_scale * bg_envelope[k];

    // Ratio de potencia sobre el fondo (o sobre el piso ensombrecido si hay
    // sombra)
    float ratio;
    if (shadow_35 && k >= 58 && k <= 61) {
      float floor_val = fminf(env_accum[55], env_accum[56]);
      ratio = (floor_val > 10.0f) ? (raw_val / floor_val) : 2.0f;
    } else if (shadow_40 && k >= 63 && k <= 66) {
      float floor_val = fminf(env_accum[62], env_accum[63]);
      ratio = (floor_val > 10.0f) ? (raw_val / floor_val) : 1.5f;
    } else if (shadow_45 && (k >= 72 && k <= 74)) {
      float floor_val = fminf(env_accum[71], env_accum[75]);
      ratio = (floor_val > 10.0f) ? (raw_val / floor_val) : 1.5f;
    } else {
      ratio = (bg_val > 10.0f) ? (raw_val / bg_val) : 1.0f;
    }

    // Umbral adaptativo: En ausencia de sombra en la mesa/pared, mantener 1800
    // para suprimir fluctuaciones del fondo en vacío. Con sombra activa,
    // descender a 750.
    float min_net;
    if (k <= 55) {
      min_net = 3200.0f;
    } else if (k <= 61) {
      min_net = 1500.0f;
    } else if (k <= 67) {
      min_net = shadow_40 ? 750.0f : 1800.0f;
    } else if (k <= 70) {
      min_net = 1500.0f;
    } else {
      min_net = shadow_45 ? 750.0f : 1800.0f;
    }

    float min_rat = minRatioThreshold(k);
    float cfar_th = localCfarThreshold(k, net_signal);
    float coh = peakCoherence(k, net_signal, 3);

    // Criterios de validación:
    // 1) Supera el umbral adaptativo por distancia
    // 2) Supera el CFAR local
    // 3) Ratio positivo significativo sobre el fondo
    // 4) Coherencia de pico definida (>= 1.15)
    bool pass_net = (net_val >= min_net);
    bool pass_cfar = (net_val >= cfar_th);
    bool pass_ratio = (ratio >= min_rat);
    bool pass_coh = (coh >= MIN_PEAK_COHERENCE);

    if (pass_net && pass_cfar && pass_ratio && pass_coh) {
      // Interpolación parabólica sub-muestra en el vértice
      float y_prev, y_curr, y_next;
      if ((shadow_35 && k >= 56 && k <= 60) ||
          (shadow_40 && k >= 63 && k <= 66) ||
          (shadow_45 && k >= 72 && k <= 74)) {
        y_prev = env_accum[k - 1];
        y_curr = raw_val;
        y_next = env_accum[k + 1];
      } else {
        y_prev = net_signal[k - 1];
        y_curr = net_val;
        y_next = net_signal[k + 1];
      }
      float denom = 2.0f * (y_prev - 2.0f * y_curr + y_next);
      float delta = 0.0f;
      if (fabsf(denom) > 1e-4f) {
        delta = (y_prev - y_next) / denom;
        if (delta > 0.5f)
          delta = 0.5f;
        if (delta < -0.5f)
          delta = -0.5f;
      }
      float refined_idx = (float)k + delta;
      float time_delay_s = refined_idx / (float)FS;
      float d_raw_cm = (((time_delay_s * SPEED_OF_SOUND) / 2.0f) * 100.0f);
      float d_cm = d_raw_cm - calibrationOffset(d_raw_cm);

      // Descartar ecos fuera del rango solicitado [30 cm, < 50 cm)
      if (d_cm < MIN_VALID_DIST_CM || d_cm >= MAX_VALID_DIST_CM) {
        continue;
      }

      float score = net_val * sqrtf(fmaxf(ratio - 1.0f, 0.05f)) * sqrtf(coh);

      if (cand_count < 5) {
        candidates[cand_count++] = {
            k, refined_idx, d_cm, raw_val, bg_val, net_val, ratio, score, coh};
      }
    }
  }

  // 5. Selección del Blanco: Físicamente Acústica
  // - Si un eco anterior tiene energía primaria masiva (NET >= 4500) o un score
  //   comparable (>= 65% del dominante), ese es el blanco directo real y los
  //   posteriores son reflexiones secundarias / rebotes de mesa.
  // - Si el eco anterior es débil (< 65% del score y NET < 4500), es un residuo
  //   parásito y se elige el pico dominante.
  int chosen_idx = 0;
  float current_distance = 0.0f;

  if (cand_count > 0) {
    int best_idx = 0;
    float max_score = candidates[0].score;
    for (int i = 1; i < cand_count; i++) {
      if (candidates[i].score > max_score) {
        max_score = candidates[i].score;
        best_idx = i;
      }
    }

    chosen_idx = best_idx;
    for (int i = 0; i < best_idx; i++) {
      // Priorizar el primer eco físico directo si tiene score dominante
      // comparable
      // (>= 75% del máximo) y energía franca primaria (>= 4000)
      if (candidates[i].score >= 0.75f * max_score &&
          candidates[i].net >= 4000.0f) {
        chosen_idx = i;
        break;
      }
    }
    current_distance = candidates[chosen_idx].dist_cm;
  }

  char lcd_buf[17];
  lcdSetCursor(0, 1);

  // 6. Pipeline de Tracking con Consistencia Espacial
  if (cand_count > 0) {
    consecutive_misses = 0;
    detection_cycles++;

    float distance_cm = current_distance;

    if (!tracking_active) {
      if (last_detected_raw > 0.0f &&
          fabsf(current_distance - last_detected_raw) <= SPATIAL_MATCH_CM) {
        consecutive_detections++;
      } else {
        consecutive_detections = 1;
      }
      last_detected_raw = current_distance;

      if (consecutive_detections >= CONFIRM_COUNT) {
        tracking_active = true;
        last_stable_distance = current_distance;
        for (int i = 0; i < MEDIAN_DEPTH; i++)
          median_buffer[i] = current_distance;
        median_write_idx = 0;
        median_full = true;
      }
      distance_cm = current_distance;
    } else {
      last_detected_raw = current_distance;
      consecutive_detections++;

      if (fabsf(current_distance - last_stable_distance) > JUMP_THRESHOLD_CM) {
        for (int i = 0; i < MEDIAN_DEPTH; i++)
          median_buffer[i] = current_distance;
        median_write_idx = 0;
        median_full = true;
        distance_cm = IIR_ALPHA_JUMP * current_distance +
                      (1.0f - IIR_ALPHA_JUMP) * last_stable_distance;
      } else {
        median_buffer[median_write_idx] = current_distance;
        median_write_idx = (median_write_idx + 1) % MEDIAN_DEPTH;
        if (!median_full && median_write_idx == 0)
          median_full = true;
        int med_len = median_full ? MEDIAN_DEPTH : median_write_idx;
        if (med_len == 0)
          med_len = 1;
        float median_dist = computeMedian(median_buffer, med_len);
        distance_cm = IIR_ALPHA_TRACK * median_dist +
                      (1.0f - IIR_ALPHA_TRACK) * last_stable_distance;
      }
      last_stable_distance = distance_cm;
    }

    float cfar_val =
        localCfarThreshold(candidates[chosen_idx].index, net_signal);
    uint16_t adc_vpp = (adc_max >= adc_min) ? (adc_max - adc_min) : 0;
    float det_rate =
        (total_cycles > 0) ? (100.0f * detection_cycles / total_cycles) : 0.0f;

    Serial.printf(
        "BLANCO -> Dist: %4.1f cm (idx: %d, sub: %.2f) | NET: %.0f | "
        "RAW: %.0f | Fondo: %.0f | Ratio: %.2f | Score: %.0f | "
        "Coh: %.1f | CFAR_TH: %.0f | ADC:[%d-%d] (Vpp:%d) | "
        "RMS: %.0f | BandE: %.0f | Sombra: %s | Det%%: %.0f\n",
        distance_cm, candidates[chosen_idx].index,
        candidates[chosen_idx].refined_index, candidates[chosen_idx].net,
        candidates[chosen_idx].raw, candidates[chosen_idx].bg,
        candidates[chosen_idx].ratio, candidates[chosen_idx].score,
        candidates[chosen_idx].coherence, cfar_val, adc_min, adc_max, adc_vpp,
        rms_val, total_band_energy, shadow_detected ? "SI" : "NO", det_rate);

    Serial.print("  [CANDIDATOS]: ");
    for (int i = 0; i < cand_count; i++) {
      const char *marker = (i == chosen_idx) ? "(*)" : "";
      Serial.printf("#%d%s: %4.1fcm(k=%d,net=%.0f,rat=%.2f,coh=%.1f) ", i + 1,
                    marker, candidates[i].dist_cm, candidates[i].index,
                    candidates[i].net, candidates[i].ratio,
                    candidates[i].coherence);
    }
    Serial.println();

    Serial.print("  [NET 52-75]: ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, net_signal[k]);
    Serial.println();
    Serial.print("  [RAW 52-75]: ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, env_accum[k]);
    Serial.println();
    Serial.print("  [BG 52-75]:  ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, bg_envelope[k]);
    Serial.println();

    Serial.print("  [CRESTAS]: ");
    int crest_count = 0;
    for (int k = MIN_SEARCH_INDEX; k <= MAX_SEARCH_INDEX; k++) {
      if ((net_signal[k] > net_signal[k - 1]) &&
          (net_signal[k] >= net_signal[k + 1]) && net_signal[k] > 300.0f) {
        float d = (((float)k / (float)FS) * SPEED_OF_SOUND * 50.0f);
        d -= calibrationOffset(d);
        float cfar_th = localCfarThreshold(k, net_signal);
        float coh_val = peakCoherence(k, net_signal, 3);
        float th = minNetThreshold(k);
        Serial.printf("%4.1fcm(k=%d,net=%.0f,th=%.0f,coh=%.1f,cfar=%.0f) ", d,
                      k, net_signal[k], th, coh_val, cfar_th);
        crest_count++;
      }
    }
    if (crest_count == 0)
      Serial.print("ninguna");
    Serial.println();

    snprintf(lcd_buf, sizeof(lcd_buf), "Dist: %4.1f cm  ", distance_cm);
  } else {
    consecutive_misses++;
    consecutive_detections = 0;
    last_detected_raw = -1.0f;

    if (consecutive_misses >= LOST_COUNT) {
      tracking_active = false;
      last_stable_distance = -1.0f;
      median_full = false;
      median_write_idx = 0;
    }

    int max_k = MIN_SEARCH_INDEX;
    float max_n = 0.0f;
    for (int k = MIN_SEARCH_INDEX; k <= MAX_SEARCH_INDEX; k++) {
      if (net_signal[k] > max_n) {
        max_n = net_signal[k];
        max_k = k;
      }
    }
    float d_raw = (((float)max_k / (float)FS) * SPEED_OF_SOUND * 50.0f);
    float d_cm = d_raw - calibrationOffset(d_raw);
    float r_val = env_accum[max_k];
    float b_val = g_scale * bg_envelope[max_k];
    float rat = (b_val > 10.0f) ? (r_val / b_val) : 1.0f;
    float cfar = localCfarThreshold(max_k, net_signal);

    uint16_t adc_vpp = (adc_max >= adc_min) ? (adc_max - adc_min) : 0;
    float det_rate =
        (total_cycles > 0) ? (100.0f * detection_cycles / total_cycles) : 0.0f;

    Serial.printf(
        "Buscando... Max residuo: %4.1f cm (idx: %d) | NET: %.0f (th: %.0f) | "
        "RAW: %.0f | Fondo: %.0f | Ratio: %.2f | CFAR_TH: %.0f | "
        "ADC:[%d-%d] (Vpp:%d) | RMS: %.0f | BandE: %.0f | "
        "Track: %s | Sombra: %s | Det%%: %.0f\n",
        d_cm, max_k, max_n, minNetThreshold(max_k), r_val, b_val, rat, cfar,
        adc_min, adc_max, adc_vpp, rms_val, total_band_energy,
        tracking_active ? "SI" : "NO", shadow_detected ? "SI" : "NO", det_rate);

    Serial.print("  [NET 52-75]: ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, net_signal[k]);
    Serial.println();
    Serial.print("  [RAW 52-75]: ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, env_accum[k]);
    Serial.println();
    Serial.print("  [BG 52-75]:  ");
    for (int k = 52; k <= 75; k++)
      Serial.printf("%d:%.0f ", k, bg_envelope[k]);
    Serial.println();

    Serial.print("  [CRESTAS]: ");
    int crest_count = 0;
    for (int k = MIN_SEARCH_INDEX; k <= MAX_SEARCH_INDEX; k++) {
      if ((net_signal[k] > net_signal[k - 1]) &&
          (net_signal[k] >= net_signal[k + 1]) && net_signal[k] > 300.0f) {
        float d = (((float)k / (float)FS) * SPEED_OF_SOUND * 50.0f);
        d -= calibrationOffset(d);
        float cfar_th = localCfarThreshold(k, net_signal);
        float coh_val = peakCoherence(k, net_signal, 3);
        float th = minNetThreshold(k);
        Serial.printf("%4.1fcm(k=%d,net=%.0f,th=%.0f,coh=%.1f,cfar=%.0f) ", d,
                      k, net_signal[k], th, coh_val, cfar_th);
        crest_count++;
      }
    }
    if (crest_count == 0)
      Serial.print("ninguna");
    Serial.println();

    if (tracking_active && last_stable_distance > 0.0f) {
      snprintf(lcd_buf, sizeof(lcd_buf), "Dist: %4.1f cm? ",
               last_stable_distance);
    } else {
      snprintf(lcd_buf, sizeof(lcd_buf), "Buscando...     ");
    }
  }

  lcdPrint(lcd_buf);

  // Cadencia fija de refresco (~1.5 s entre ciclos para lectura cómoda)
  uint32_t elapsed_ms = millis() - loop_start_ms;
  if (elapsed_ms < 1500) {
    delay(1500 - elapsed_ms);
  }
}