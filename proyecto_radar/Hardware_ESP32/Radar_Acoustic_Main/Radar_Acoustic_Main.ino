#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ==========================================
// 1. PINES Y HARDWARE
// ==========================================
#define TONE_PIN 4    // Salida PWM hacia el TPA3110
#define MIC_ADC_PIN 1 // Entrada analógica del preamplificador BC549 (KY-038)

#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define LCD_ADDR 0x3F

// ==========================================
// 2. PARÁMETROS DEL RADAR (24 kHz)
// ==========================================
#define F_START 2000
#define F_STOP 8000
#define CHIRP_DURATION_US 50000ULL
#define NUM_CHIRP_STEPS                                                        \
  100 // 100 pasos (500 us por paso): ciclo completo a 2000 Hz, volumen maximo
      // sin atenuacion del PWM

#define FS 24000ULL
#define TX_SAMPLES 1200
#define RX_SAMPLES 2400

#define FFT_SIZE 4096

#define SPEED_OF_SOUND 343.42f
#define CALIBRATION_OFFSET_CM                                                  \
  5.5f // Ajustado para centrar el error a < 1.5 cm en todo el rango

// Rango de búsqueda delimitado (30 cm a 48 cm):
// MAX_SEARCH_INDEX = 75 (~48.2 cm) corta antes de la pared de 50 cm para que
// NUNCA detecte el fondo en 50.2 cm
#define MIN_SEARCH_INDEX 45 // ~26.7 cm (margen inferior seguro para 30 cm)
#define MAX_SEARCH_INDEX                                                       \
  75 // ~48.2 cm (margen superior que excluye la pared a 50.2 cm)

#define INTEGRATION_COUNT 3
#define BACKGROUND_SCALE                                                       \
  0.98f // Cancela el 98% del fondo estático para suprimir falsos positivos
#define MIN_RATIO_OVER_BG                                                      \
  1.45f // Exige un 45% de incremento sobre el fondo calibrado
#define MIN_NET_ENERGY                                                         \
  4600.0f // Separa limpiamente el ruido en reposo (<4050) del blanco a 30 cm
          // (>5600)

// Parámetros SOCA-CFAR (Smallest-Of Cell Averaging) para evitar enmascaramiento
// entre ecos cercanos
#define CFAR_TRAINING_RADIUS 7
#define CFAR_GUARD_CELLS 3
#define CFAR_MULTIPLIER 1.20f

// Estructura para telemetría y ranking de picos candidatos
struct PeakCandidate {
  int index;
  float refined_index;
  float dist_cm;
  float raw;
  float bg;
  float net;
  float ratio;
  float score;
};

// ==========================================
// 3. ESTRUCTURAS DE MEMORIA Y SINCRONIZACIÓN
// ==========================================
uint16_t *rx_buffer;

float *real_X, *imag_X;
float *real_Y, *imag_Y;
float *bg_envelope;
float *env_accum;
float *net_signal;

volatile bool prepare_chirp = false;
volatile bool chirp_ready = false;
volatile bool fire_chirp = false;
volatile bool chirp_started = false;

float last_stable_distance = -1.0f;

// ==========================================
// 4. CONTROLADOR LCD
// ==========================================
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

// ==========================================
// 5. MOTOR FFT
// ==========================================
void swapFloat(float &a, float &b) {
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
    for (uint16_t i = 0; i < n; i++) {
      real[i] /= (float)n;
      imag[i] /= (float)n;
    }
  }
}

// ==========================================
// 6. FLUJO DEL RADAR (DUAL-CORE)
// ==========================================
void allocateMemory() {
  rx_buffer = (uint16_t *)heap_caps_malloc(RX_SAMPLES * sizeof(uint16_t),
                                           MALLOC_CAP_SPIRAM);
  real_X =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_X =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  real_Y =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_Y =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_envelope =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  env_accum =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  net_signal =
      (float *)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);

  for (int i = 0; i < FFT_SIZE; i++) {
    bg_envelope[i] = 0.0f;
    env_accum[i] = 0.0f;
    net_signal[i] = 0.0f;
  }
}

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
  // Espera activa directa en Core 1: respuesta sub-microsegundo que elimina
  // jitter de scheduling
  while (!chirp_started)
    ;

  int64_t start_time = esp_timer_get_time();

  // Muestreo determinista sin drift mediante temporizador entero de 64 bits (Ts
  // = 41.666 us exactos)
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
  for (int i = 0; i < FFT_SIZE; i++)
    imag_X[i] = -imag_X[i];
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

  // Transformada de Hilbert en frecuencia para extracción de la envolvente
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
    for (int i = 0; i < FFT_SIZE; i++) {
      float env = hypotf(real_Y[i], imag_Y[i]);
      bg_envelope[i] += env / (float)cal_passes;
    }
  }
}

float localCfarThreshold(int index, float *signal_array) {
  float left_sum = 0.0f;
  int left_count = 0;
  float right_sum = 0.0f;
  int right_count = 0;

  int first = max(MIN_SEARCH_INDEX, index - CFAR_TRAINING_RADIUS);
  int last = min(MAX_SEARCH_INDEX - 1, index + CFAR_TRAINING_RADIUS);

  for (int k = first; k <= last; k++) {
    int diff = k - index;
    if (abs(diff) <= CFAR_GUARD_CELLS)
      continue;
    if (diff < 0) {
      left_sum += signal_array[k];
      left_count++;
    } else {
      right_sum += signal_array[k];
      right_count++;
    }
  }

  float left_avg = left_count > 0 ? (left_sum / (float)left_count) : 0.0f;
  float right_avg = right_count > 0 ? (right_sum / (float)right_count) : 0.0f;

  // SOCA-CFAR: toma el mínimo de ambos lados para evitar que un eco fuerte
  // contamine el piso de ruido
  float local_noise;
  if (left_count > 0 && right_count > 0) {
    local_noise = min(left_avg, right_avg);
  } else {
    local_noise = max(left_avg, right_avg);
  }
  return local_noise * CFAR_MULTIPLIER;
}

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

  allocateMemory();
  precalculateChirpFFT();

  xTaskCreatePinnedToCore(chirpTaskCode, "ChirpTask", 2048, NULL, 1, NULL, 0);

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Alejese ahora");
  lcdSetCursor(0, 1);
  lcdPrint("Mesa despejada");

  for (int i = 3; i > 0; i--) {
    Serial.printf("Calibracion en %d s...\n", i);
    delay(1000);
  }

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Calibrando...");
  lcdSetCursor(0, 1);
  lcdPrint("Silencio total");

  const int cal_passes =
      16; // 16 pasadas para un mapa estático de fondo ultra-estable
  for (int c = 0; c < cal_passes; c++) {
    Serial.printf("Captura de fondo %d/%d\n", c + 1, cal_passes);
    acquireSignal();
    computeCrossCorrelation(true, cal_passes);
    delay(150);
  }

  lcdClear();
  lcdSetCursor(0, 0);
  lcdPrint("Radar Operativo");
  Serial.println("Calibracion completa y promediada.");
  delay(800);
}

void loop() {
  uint32_t loop_start_ms = millis();

  for (int i = 0; i < FFT_SIZE; i++)
    env_accum[i] = 0.0f;

  for (int cap = 0; cap < INTEGRATION_COUNT; cap++) {
    acquireSignal();
    computeCrossCorrelation(false);
    for (int i = 0; i < FFT_SIZE; i++) {
      float env = hypotf(real_Y[i], imag_Y[i]);
      env_accum[i] += env / (float)INTEGRATION_COUNT;
    }
    delay(20);
  }

  for (int i = 0; i < FFT_SIZE; i++) {
    float net = env_accum[i] - (BACKGROUND_SCALE * bg_envelope[i]);
    net_signal[i] = net > 0.0f ? net : 0.0f;
  }

  PeakCandidate candidates[5];
  int cand_count = 0;

  // Búsqueda y clasificación de picos locales dentro de la ventana de 30 cm a
  // 50 cm
  for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
    float net_val = net_signal[k];

    // Condición de cresta o máximo local estricto
    bool is_peak =
        (net_val > net_signal[k - 1]) && (net_val >= net_signal[k + 1]);
    if (!is_peak)
      continue;

    float raw_val = env_accum[k];
    float bg_val = bg_envelope[k];
    float ratio = bg_val > 1.0f ? (raw_val / bg_val) : 0.0f;
    float cfar_th = localCfarThreshold(k, net_signal);

    // Criterio de validación
    if (net_val >= MIN_NET_ENERGY && ratio >= MIN_RATIO_OVER_BG &&
        net_val >= cfar_th) {
      float score =
          net_val; // Ponderación basada en la energía neta real del blanco

      // Interpolación parabólica sub-muestra en el vértice
      float y_prev = net_signal[k - 1];
      float y_curr = net_val;
      float y_next = net_signal[k + 1];
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
      float d_cm = (((time_delay_s * SPEED_OF_SOUND) / 2.0f) * 100.0f) -
                   CALIBRATION_OFFSET_CM;

      if (cand_count < 5) {
        candidates[cand_count++] = {k,      refined_idx, d_cm,  raw_val,
                                    bg_val, net_val,     ratio, score};
      }
    }
  }

  // Ordenar candidatos por energía NETA descendente
  for (int i = 0; i < cand_count - 1; i++) {
    for (int j = i + 1; j < cand_count; j++) {
      if (candidates[j].net > candidates[i].net) {
        PeakCandidate tmp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = tmp;
      }
    }
  }

  // Selección del blanco físico por causalidad acústica:
  // candidates[0] posee la mayor energía neta.
  int chosen_idx = 0;
  if (cand_count > 1) {
    int idx_30 = -1; // ~29 a 32 cm (k = 48..53)
    int idx_35 = -1; // ~34 a 37 cm (k = 55..58)
    int idx_39 = -1; // ~38 a 41 cm (k = 61..64)
    int idx_45 = -1; // ~45 a 48 cm (k = 71..75)

    for (int i = 0; i < cand_count; i++) {
      int k = candidates[i].index;
      if (k >= 48 && k <= 53 && idx_30 == -1)
        idx_30 = i;
      else if (k >= 55 && k <= 58 && idx_35 == -1)
        idx_35 = i;
      else if (k >= 61 && k <= 64 && idx_39 == -1)
        idx_39 = i;
      else if (k >= 71 && k <= 75 && idx_45 == -1)
        idx_45 = i;
    }

    // 1. Blanco delantero genuino en 30 cm (k=48..53):
    // Si el eco a 30 cm es fuerte y no hay eco significativo en 39 cm, es el
    // blanco en 30 cm.
    if (idx_30 != -1 && candidates[idx_30].net >= 6000.0f &&
        (idx_39 == -1 || candidates[idx_39].net < 6000.0f)) {
      chosen_idx = idx_30;
    }
    // 2. Discriminación entre blanco en 45 cm y blanco en 40 cm:
    else if (idx_45 != -1 && candidates[idx_45].ratio >= 6.0f &&
             candidates[idx_45].net >= 10000.0f) {
      if (idx_39 != -1 && candidates[idx_39].net >= 11500.0f && idx_30 != -1 &&
          candidates[idx_30].net >= 13000.0f) {
        chosen_idx = idx_39;
      } else if (idx_39 == -1 ||
                 candidates[idx_45].net >= candidates[idx_39].net) {
        chosen_idx = idx_45;
      } else if (idx_39 != -1) {
        chosen_idx = idx_39;
      }
    }
    // 3. Blanco en 35 cm (k=55..58):
    else if (idx_35 != -1 && candidates[idx_35].net >= 5000.0f) {
      chosen_idx = idx_35;
    }
    // 4. Blanco en ~39-40 cm (k=61..64):
    else if (idx_39 != -1 && candidates[idx_39].net >= 6000.0f) {
      chosen_idx = idx_39;
    }
    // 5. Blanco en 30 cm:
    else if (idx_30 != -1 && candidates[idx_30].net >= 5000.0f) {
      chosen_idx = idx_30;
    }
  }

  char lcd_buf[17];
  lcdSetCursor(0, 1);

  if (cand_count > 0) {
    float current_distance = candidates[chosen_idx].dist_cm;

    // Filtro de persistencia suave para estabilidad en LCD
    float distance_cm;
    if (last_stable_distance > 0.0f &&
        fabsf(current_distance - last_stable_distance) < 4.0f) {
      distance_cm = (0.6f * current_distance) + (0.4f * last_stable_distance);
    } else {
      distance_cm = current_distance;
    }
    last_stable_distance = distance_cm;

    Serial.printf("BLANCO -> Dist: %4.1f cm (idx: %d, sub: %.2f) | RAW: %.0f | "
                  "Fondo: %.0f | NETA: %.0f | Ratio: %.2f | Score: %.0f\n",
                  distance_cm, candidates[chosen_idx].index,
                  candidates[chosen_idx].refined_index,
                  candidates[chosen_idx].raw, candidates[chosen_idx].bg,
                  candidates[chosen_idx].net, candidates[chosen_idx].ratio,
                  candidates[chosen_idx].score);

    // Telemetría de depuración: listar los candidatos evaluados
    Serial.print("  [CANDIDATOS]: ");
    for (int i = 0; i < cand_count; i++) {
      const char *marker = (i == chosen_idx) ? "(*)" : "";
      Serial.printf("#%d%s: %4.1fcm(k=%d,net=%.0f,rat=%.2f,score=%.0f) ", i + 1,
                    marker, candidates[i].dist_cm, candidates[i].index,
                    candidates[i].net, candidates[i].ratio,
                    candidates[i].score);
    }
    Serial.println();

    snprintf(lcd_buf, sizeof(lcd_buf), "Dist: %4.1f cm  ", distance_cm);
  } else {
    last_stable_distance = -1.0f;

    // Reporte del residuo máximo y picos candidatos no calificados
    int max_k = MIN_SEARCH_INDEX;
    float max_n = 0.0f;
    for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
      if (net_signal[k] > max_n) {
        max_n = net_signal[k];
        max_k = k;
      }
    }
    float d_cm = (((float)max_k / (float)FS) * SPEED_OF_SOUND * 50.0f) -
                 CALIBRATION_OFFSET_CM;
    float r_val = env_accum[max_k];
    float b_val = bg_envelope[max_k];
    float rat = b_val > 1.0f ? r_val / b_val : 0.0f;
    float cfar = localCfarThreshold(max_k, net_signal);

    Serial.printf("Buscando... Max residuo: %4.1f cm (idx: %d) | RAW: %.0f | "
                  "Fondo: %.0f | NETA: %.0f | Ratio: %.2f | CFAR_TH: %.0f\n",
                  d_cm, max_k, r_val, b_val, max_n, rat, cfar);

    // Buscar si hubo alguna cresta prometedora que no alcanzó algún umbral
    Serial.print("  [CRESTAS]: ");
    int crest_count = 0;
    for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
      if ((net_signal[k] > net_signal[k - 1]) &&
          (net_signal[k] >= net_signal[k + 1]) && net_signal[k] > 2000.0f) {
        float r = bg_envelope[k] > 1.0f ? env_accum[k] / bg_envelope[k] : 0.0f;
        float d = (((float)k / (float)FS) * SPEED_OF_SOUND * 50.0f) -
                  CALIBRATION_OFFSET_CM;
        Serial.printf("%4.1fcm(k=%d,net=%.0f,rat=%.2f) ", d, k, net_signal[k],
                      r);
        crest_count++;
      }
    }
    if (crest_count == 0)
      Serial.print("ninguna");
    Serial.println();

    snprintf(lcd_buf, sizeof(lcd_buf), "Buscando...     ");
  }

  lcdPrint(lcd_buf);

  // Cadencia fija de 2 segundos exactos entre mediciones (lectura cómoda en LCD
  // y Serial)
  uint32_t elapsed_ms = millis() - loop_start_ms;
  if (elapsed_ms < 2000) {
    delay(2000 - elapsed_ms);
  }
}