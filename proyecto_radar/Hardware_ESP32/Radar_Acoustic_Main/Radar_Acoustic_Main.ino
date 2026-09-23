#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <FS.h>
#include <LittleFS.h>

// ==========================================
// 1. PINES Y HARDWARE
// ==========================================
#define TONE_PIN 4
#define MIC_ADC_PIN 1

#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define LCD_ADDR 0x3F

// ==========================================
// 2. PARÁMETROS DEL RADAR (24kHz)
// ==========================================
#define F_START 2000
#define F_STOP  8000
#define CHIRP_DURATION_US 50000ULL   
#define NUM_CHIRP_STEPS 100          

#define FS 24000ULL                  
#define TX_SAMPLES 1200              
#define RX_SAMPLES 2400              

#define FFT_SIZE 4096                // Potencia de dos para la FFT radix-2.

#define SPEED_OF_SOUND 343.42f       // Velocidad del sonido en m/s.
#define CALIBRATION_OFFSET_CM 2.9f   

#define MIN_SEARCH_INDEX 45          // ~28 cm, cerca del limite de 30 cm
#define MAX_SEARCH_INDEX 125         // ~86 cm, limite superior de la prueba
#define MIN_PEAK_SCORE 0.025f        // Score absoluto minimo del eco
#define NOISE_SIGMA_FACTOR 2.5f       // Margen sobre la variacion del fondo
#define MIN_BACKGROUND_EXCESS 0.010f
#define PEAK_PROMINENCE_FACTOR 1.10f
#define PROMINENCE_GUARD_SAMPLES 3
#define CLUTTER_MASK_SCORE 0.015f
#define CLUTTER_GUARD_SAMPLES 5
#define CONFIRMATION_SAMPLES 2
#define TRACK_LOST_SAMPLES 2
#define RESULT_HOLD_MS 3000UL
#define MEASUREMENT_LOG_FILE "/radar.csv"

// ==========================================
// 3. ESTRUCTURAS DE MEMORIA Y SINCRONIZACIÓN
// ==========================================
uint16_t *rx_buffer;       

float *real_X, *imag_X;    
float *real_Y, *imag_Y;    
float *bg_envelope;        
float *bg_second_moment;
float *bg_score;
float *bg_score_second_moment;
bool *clutter_mask;

volatile bool prepare_chirp = false;
volatile bool chirp_ready = false;
volatile bool fire_chirp = false;
volatile bool chirp_started = false;

float tx_energy = 0.0f;
float last_dc_offset = 0.0f;
uint32_t measurement_number = 0;
bool log_available = false;
int pending_index = -1;
uint8_t pending_count = 0;
bool track_confirmed = false;
int tracked_index = -1;
uint8_t track_missed_count = 0;

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
void lcdPrint(const char* str) { while (*str) lcdWriteChar(*str++); }
void lcdSetCursor(uint8_t col, uint8_t row) {
  int row_offsets[] = { 0x00, 0x40 };
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
  send4Bits(0x30, 0); delayMicroseconds(4500);
  send4Bits(0x30, 0); delayMicroseconds(4500);
  send4Bits(0x30, 0); delayMicroseconds(150);
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
  float temp = a; a = b; b = temp;
}

void computeFFT(float* real, float* imag, uint16_t n, bool inverse) {
  // FFT radix-2 de Cooley-Tukey. Calcula la misma transformada que la DFT
  // directa, pero con complejidad O(N log2(N)) en lugar de O(N^2).
  uint16_t j = 0;
  for (uint16_t i = 0; i < n - 1; i++) {
    if (i < j) {
      swapFloat(real[i], real[j]);
      swapFloat(imag[i], imag[j]);
    }
    uint16_t m = n >> 1;
    while (j >= m) { j -= m; m >>= 1; }
    j += m;
  }

  // Cada etapa combina subtransformadas mediante operaciones mariposa.
  for (uint16_t step = 1; step < n; step <<= 1) {
    // La FFT usa exp(-j*2*pi*k*n/N); la IFFT cambia el signo.
    float dir = inverse ? 1.0 : -1.0;
    float theta = dir * PI / step;
    float w_real = cos(theta);
    float w_imag = sin(theta);

    for (uint16_t i = 0; i < n; i += (step << 1)) {
      float u_real = 1.0;
      float u_imag = 0.0;
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

  // La IFFT debe normalizarse por N para recuperar la amplitud temporal.
  if (inverse) {
    for (uint16_t i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

// ==========================================
// 6. FLUJO DEL RADAR (DUAL-CORE SINCRONIZADO)
// ==========================================
void allocateMemory() {
  // Los buffers de FFT y recepcion se reservan en PSRAM para conservar
  // la RAM interna del ESP32 para tareas del sistema.
  rx_buffer = (uint16_t*) heap_caps_malloc(RX_SAMPLES * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
  real_X = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_X = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  real_Y = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_Y = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_envelope = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_second_moment = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_score = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_score_second_moment = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  clutter_mask = (bool*) heap_caps_malloc(FFT_SIZE * sizeof(bool), MALLOC_CAP_SPIRAM);

  for (int i = 0; i < FFT_SIZE; i++) {
    bg_envelope[i] = 0.0f;
    bg_second_moment[i] = 0.0f;
    bg_score[i] = 0.0f;
    bg_score_second_moment[i] = 0.0f;
    clutter_mask[i] = false;
  }
}

void buildClutterMask() {
  // Un maximo persistente durante la calibracion sin objeto indica una
  // reflexion del montaje. Se conserva para diagnostico como clutter.
  for (int k = MIN_SEARCH_INDEX + CLUTTER_GUARD_SAMPLES;
       k < MAX_SEARCH_INDEX - CLUTTER_GUARD_SAMPLES; k++) {
    bool local_maximum = true;
    for (int offset = 1; offset <= CLUTTER_GUARD_SAMPLES; offset++) {
      if (bg_score[k] < bg_score[k - offset] ||
          bg_score[k] < bg_score[k + offset]) {
        local_maximum = false;
        break;
      }
    }
    if (local_maximum && bg_score[k] >= CLUTTER_MASK_SCORE) {
      for (int offset = -CLUTTER_GUARD_SAMPLES;
           offset <= CLUTTER_GUARD_SAMPLES; offset++) {
        clutter_mask[k + offset] = true;
      }
      Serial.print("CLUTTER,retardo=");
      Serial.print(k);
      Serial.print(",distancia_cm=");
      Serial.println((((float)k / FS) * SPEED_OF_SOUND * 100.0f / 2.0f) -
                     CALIBRATION_OFFSET_CM, 1);
    }
  }
}

void chirpTaskCode(void * pvParameters) {
  uint32_t step_delay_us = CHIRP_DURATION_US / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;
  
  while(true) {
    if (prepare_chirp) {
      prepare_chirp = false;
      chirp_ready = true; 
      
      while(!fire_chirp) { } 
      fire_chirp = false;
      
      // El chirp real es escalonado: cada tono permanece 500 us. La
      // referencia matematica se construye con exactamente estos pasos.
      int64_t start_t = esp_timer_get_time();
      ledcWriteTone(TONE_PIN, F_START);
      chirp_started = true;
      for(int i = 0; i < NUM_CHIRP_STEPS; i++) {
        if (i > 0) {
          ledcWriteTone(TONE_PIN, F_START + (uint32_t)(i * freq_step));
        }
        int64_t target = start_t + (i + 1) * step_delay_us;
        while(esp_timer_get_time() < target);
      }
      ledcWriteTone(TONE_PIN, 0);
    }
    vTaskDelay(1); 
  }
}

void acquireSignal() {
  float sample_interval_us = 1000000.0f / FS; 
  
  prepare_chirp = true;
  while(!chirp_ready) { delay(1); } 
  chirp_ready = false;
  
  chirp_started = false;
  fire_chirp = true; 
  while(!chirp_started) { taskYIELD(); }
  
  int64_t start_time = esp_timer_get_time();
  float next_sample_time = (float)start_time;

  // Muestreo uniforme: Ts = 1/FS = 41.67 us para FS = 24 kHz.
  for (int i = 0; i < RX_SAMPLES; i++) {
    int64_t current_time = esp_timer_get_time();
    while (current_time < (int64_t)next_sample_time) {
      current_time = esp_timer_get_time();
    }
    next_sample_time += sample_interval_us;
    
    rx_buffer[i] = analogRead(MIC_ADC_PIN);
  }
}

void precalculateChirpFFT() {
  for (int i = 0; i < FFT_SIZE; i++) { real_X[i] = 0.0f; imag_X[i] = 0.0f; }
  const int samples_per_step = TX_SAMPLES / NUM_CHIRP_STEPS;
  const float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;
  float phase = 0.0f;
  tx_energy = 0.0f;
  
  // x[n] es la referencia del chirp transmitido. Se guarda invertida
  // en tiempo porque la correlacion puede expresarse como convolucion:
  // r[n] = y[n] * reverse(x[n]).
  for (int n = 0; n < TX_SAMPLES; n++) {
    float sample = sin(phase);
    real_X[TX_SAMPLES - 1 - n] = sample;
    tx_energy += sample * sample;
    
    int step = n / samples_per_step;
    if (step >= NUM_CHIRP_STEPS) step = NUM_CHIRP_STEPS - 1;
    float frequency = F_START + step * freq_step;
    phase += 2.0f * PI * frequency / (float)FS;
  }
  
  // Xc[k] se usara en la multiplicacion FFT de la correlacion.
  computeFFT(real_X, imag_X, FFT_SIZE, false);
}

// Calcula la correlacion mediante FFT e IFFT. Con y[n] como recepcion y
// xc[n] = reverse(x[n]) como referencia:
// r[n] = IFFT(FFT(y[n]) * FFT(xc[n])).
// El zero-padding evita que la operacion sea circular dentro de la ventana.
void computeCrossCorrelation(bool is_calibration, int cal_count = 1) {
  // Elimina el nivel DC del ADC antes de pasar la señal al dominio
  // frecuencial; de lo contrario el offset genera energia artificial.
  float dc_sum = 0.0f;
  for (int i = 0; i < RX_SAMPLES; i++) {
    dc_sum += (float)rx_buffer[i];
  }
  last_dc_offset = dc_sum / (float)RX_SAMPLES;

  for (int i = 0; i < FFT_SIZE; i++) {
    if (i < RX_SAMPLES) {
      real_Y[i] = (float)rx_buffer[i] - last_dc_offset;
    } else {
      real_Y[i] = 0.0f;
    }
    imag_Y[i] = 0.0f;
  }

  // FFT de la recepcion: y[n] -> Y[k].
  computeFFT(real_Y, imag_Y, FFT_SIZE, false);

  // Producto espectral. Por la propiedad de convolucion, este producto
  // equivale a correlacionar la recepcion con el chirp transmitido.
  for (int i = 0; i < FFT_SIZE; i++) {
    float r = real_Y[i] * real_X[i] - imag_Y[i] * imag_X[i];
    float im = real_Y[i] * imag_X[i] + imag_Y[i] * real_X[i];
    real_Y[i] = r;
    imag_Y[i] = im;
  }

  // IFFT: devuelve la respuesta del filtro adaptado al dominio temporal.
  computeFFT(real_Y, imag_Y, FFT_SIZE, true);

  for (int i = 0; i < FFT_SIZE; i++) {
    // Se usa magnitud porque la fase del eco puede cambiar el signo del
    // valor real aunque la energia del eco siga siendo significativa.
    float current_envelope = abs(real_Y[i]);
    real_Y[i] = current_envelope;
  }

  if (is_calibration) {
    for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
      float rx_energy = 0.0f;
      for (int n = 0; n < TX_SAMPLES; n++) {
        float sample = (float)rx_buffer[k + n] - last_dc_offset;
        rx_energy += sample * sample;
      }
      float denominator = sqrtf(tx_energy * rx_energy);
      int correlation_index = k + TX_SAMPLES - 1;
      float score = denominator > 1.0f ? real_Y[correlation_index] / denominator : 0.0f;
      bg_score[k] += score / (float)cal_count;
      bg_score_second_moment[k] += (score * score) / (float)cal_count;
    }
  }
}

void initializeMeasurementLog() {
  log_available = LittleFS.begin(false);
  if (!log_available) {
    Serial.println("WARN,LittleFS corrupto; se intentara formatear la particion");
    LittleFS.end();
    log_available = LittleFS.begin(true);
    if (!log_available) {
      Serial.println("WARN,LittleFS no disponible; las mediciones solo saldran por Serial");
      return;
    }
  }

  if (!LittleFS.exists(MEASUREMENT_LOG_FILE)) {
    File log_file = LittleFS.open(MEASUREMENT_LOG_FILE, FILE_WRITE);
    if (log_file) {
      log_file.println("medicion,distancia_cm,score,umbral,detectado,pico_idx,exceso,segundo_exceso,candidato");
      log_file.close();
    }
  }

  File log_file = LittleFS.open(MEASUREMENT_LOG_FILE, FILE_READ);
  if (log_file) {
    uint32_t stored_rows = 0;
    while (log_file.available()) {
      if (log_file.read() == '\n') stored_rows++;
    }
    log_file.close();
    measurement_number = stored_rows > 0 ? stored_rows - 1 : 0;
  }
  Serial.print("LOG_READY,mediciones_previas=");
  Serial.println(measurement_number);
}

void appendMeasurement(float distance_cm, float score, float threshold,
                       bool detected, int peak_index, float excess,
                       float second_excess, bool candidate) {
  if (!log_available) return;

  File log_file = LittleFS.open(MEASUREMENT_LOG_FILE, FILE_APPEND);
  if (!log_file) {
    Serial.println("WARN,no se pudo escribir radar.csv");
    return;
  }
  log_file.print(measurement_number);
  log_file.print(",");
  log_file.print(distance_cm, 1);
  log_file.print(",");
  log_file.print(score, 4);
  log_file.print(",");
  log_file.print(threshold, 4);
  log_file.print(",");
  log_file.print(detected ? 1 : 0);
  log_file.print(",");
  log_file.print(peak_index);
  log_file.print(",");
  log_file.print(excess, 4);
  log_file.print(",");
  log_file.print(second_excess, 4);
  log_file.print(",");
  log_file.println(candidate ? 1 : 0);
  log_file.close();
}

void printStoredMeasurements() {
  if (!log_available) return;

  File log_file = LittleFS.open(MEASUREMENT_LOG_FILE, FILE_READ);
  if (!log_file) return;

  Serial.println("LOG_BEGIN");
  while (log_file.available()) {
    Serial.write(log_file.read());
  }
  Serial.println("LOG_END");
  log_file.close();
}

void clearMeasurementLog() {
  if (!log_available) return;

  LittleFS.remove(MEASUREMENT_LOG_FILE);
  File log_file = LittleFS.open(MEASUREMENT_LOG_FILE, FILE_WRITE);
  if (log_file) {
    log_file.println("medicion,distancia_cm,score,umbral,detectado,pico_idx,exceso,segundo_exceso,candidato");
    log_file.close();
    measurement_number = 0;
    Serial.println("LOG_CLEARED");
  } else {
    Serial.println("WARN,no se pudo borrar radar.csv");
  }
}

void processSerialCommand() {
  while (Serial.available() > 0) {
    char command = (char)Serial.read();
    if (command == 'C' || command == 'c') {
      clearMeasurementLog();
    }
  }
}

void setup() {
  Serial.begin(115200);
  initializeMeasurementLog();
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ledcAttach(TONE_PIN, F_START, 8);
  ledcWriteTone(TONE_PIN, 0);

  lcdInit();
  lcdSetCursor(0, 0); lcdPrint("Iniciando PSRAM");

  allocateMemory();
  precalculateChirpFFT();

  xTaskCreatePinnedToCore(chirpTaskCode, "ChirpTask", 2048, NULL, 1, NULL, 0);

  lcdClear();
  lcdSetCursor(0, 0); lcdPrint("Alejese ahora");
  lcdSetCursor(0, 1); lcdPrint("Mesa libre...");
  
  for(int i = 5; i > 0; i--) {
    Serial.print("Comenzando calibracion en "); Serial.println(i);
    delay(1000);
  }

  lcdClear();
  lcdSetCursor(0, 0); lcdPrint("Calibrando...");
  lcdSetCursor(0, 1); lcdPrint("Silencio total");
  
  const int cal_passes = 10;
  for (int c = 0; c < cal_passes; c++) {
    Serial.print("Muestra de calibracion "); Serial.print(c + 1);
    Serial.print("/"); Serial.println(cal_passes);
    acquireSignal();
    computeCrossCorrelation(true, cal_passes); 
    delay(1000); 
  }
  
  lcdClear();
  lcdSetCursor(0, 0); lcdPrint("Radar Listo!");
  Serial.println("Calibracion profunda completa y promediada.");
  buildClutterMask();
  printStoredMeasurements();
}

// Ciclo principal con umbral estadístico simplificado
void loop() {
  processSerialCommand();
  delay(1000);

  measurement_number++;
  acquireSignal();
  computeCrossCorrelation(false);

  const int search_count = MAX_SEARCH_INDEX - MIN_SEARCH_INDEX;
  float scores[MAX_SEARCH_INDEX - MIN_SEARCH_INDEX];
  float raw_scores[MAX_SEARCH_INDEX - MIN_SEARCH_INDEX];

  // Correlacion normalizada por retardo:
  // score[k] = |r[k]| / sqrt(E_tx * E_rx[k]).
  // Asi el score es menos sensible a la atenuacion y a la ganancia ADC.
  for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
    float rx_energy = 0.0f;
    for (int n = 0; n < TX_SAMPLES; n++) {
      float sample = (float)rx_buffer[k + n] - last_dc_offset;
      rx_energy += sample * sample;
    }
    float denominator = sqrtf(tx_energy * rx_energy);
    // La convolucion con TX_SAMPLES muestras desplaza el pico fisico
    // TX_SAMPLES - 1 posiciones en la salida de la IFFT.
    int correlation_index = k + TX_SAMPLES - 1;
    float raw_score = denominator > 1.0f ? real_Y[correlation_index] / denominator : 0.0f;
    float background_variance = bg_score_second_moment[k] -
        (bg_score[k] * bg_score[k]);
    if (background_variance < 0.0f) background_variance = 0.0f;
    // Se compara contra el fondo aprendido en el mismo retardo, mas un
    // margen estadistico basado en las capturas de calibracion.
    float background_limit = bg_score[k] +
        NOISE_SIGMA_FACTOR * sqrtf(background_variance);
    raw_scores[k - MIN_SEARCH_INDEX] = raw_score;
    scores[k - MIN_SEARCH_INDEX] = raw_score > background_limit ?
        raw_score - background_limit : 0.0f;
  }

  int best_index = -1;
  float best_score = 0.0f;
  float best_excess = 0.0f;
  
  for (int i = 0; i < search_count; i++) {
    if (scores[i] > best_excess) {
      best_excess = scores[i];
      best_score = raw_scores[i];
      best_index = MIN_SEARCH_INDEX + i;
    }
  }

  float second_best_score = 0.0f;
  for (int i = 0; i < search_count; i++) {
    int distance_from_peak = abs((MIN_SEARCH_INDEX + i) - best_index);
    if (distance_from_peak > PROMINENCE_GUARD_SAMPLES &&
        scores[i] > second_best_score) {
      second_best_score = scores[i];
    }
  }
  // El pico elegido debe sobresalir frente a otros picos alejados.
  bool prominent_peak = second_best_score <= 0.0f ||
      best_excess >= (PEAK_PROMINENCE_FACTOR * second_best_score);

    float score_threshold = MIN_PEAK_SCORE;
    if (best_index > 0) {
    float best_variance = bg_score_second_moment[best_index] -
      (bg_score[best_index] * bg_score[best_index]);
    if (best_variance < 0.0f) best_variance = 0.0f;
    score_threshold = bg_score[best_index] +
      NOISE_SIGMA_FACTOR * sqrtf(best_variance);
    if (score_threshold < MIN_PEAK_SCORE) score_threshold = MIN_PEAK_SCORE;
    }

    bool candidate_detected = best_index > 0 && best_score > score_threshold &&
      best_excess >= MIN_BACKGROUND_EXCESS && prominent_peak;
    // La confirmacion temporal evita aceptar un ruido que aparece una sola
    // vez. Dos picos cercanos consecutivos confirman el mismo eco.
    bool same_as_pending = candidate_detected && pending_index >= 0 &&
      abs(best_index - pending_index) <= PROMINENCE_GUARD_SAMPLES;

    if (same_as_pending) {
      pending_count++;
      pending_index = (pending_index + best_index) / 2;
    } else if (candidate_detected) {
      pending_index = best_index;
      pending_count = 1;
    } else if (!track_confirmed) {
      pending_index = -1;
      pending_count = 0;
    }

    if (!track_confirmed && pending_count >= CONFIRMATION_SAMPLES) {
      track_confirmed = true;
      tracked_index = pending_index;
      track_missed_count = 0;
    }

    bool detected = false;
    if (track_confirmed) {
      if (candidate_detected && abs(best_index - tracked_index) <= PROMINENCE_GUARD_SAMPLES) {
        tracked_index = (tracked_index + best_index) / 2;
        track_missed_count = 0;
        detected = true;
      } else {
        track_missed_count++;
        if (track_missed_count >= TRACK_LOST_SAMPLES) {
          track_confirmed = false;
          tracked_index = -1;
          pending_index = -1;
          pending_count = 0;
        }
      }
    }
  float distance_cm = 0.0f;
  // Radar monostatico: el eco recorre ida y vuelta.
  // tau = indice / FS, por lo que d = c*tau/2.
  if (detected) {
    float time_delay_s = (float)best_index / (float)FS;
    distance_cm = (((time_delay_s * SPEED_OF_SOUND) / 2.0f) * 100.0f) - CALIBRATION_OFFSET_CM;
  }

  Serial.print("DATA,");
  Serial.print(measurement_number);
  Serial.print(",");
  Serial.print(detected ? distance_cm : 0.0f, 1);
  Serial.print(",");
  Serial.print(best_score, 4);
  Serial.print(",");
  Serial.print(score_threshold, 4);
  Serial.print(",");
  Serial.println(detected ? 1 : 0);
  Serial.print("PEAK,idx="); Serial.print(best_index);
  Serial.print(",dist_cm=");
  Serial.print(best_index > 0 ? ((((float)best_index / FS) * SPEED_OF_SOUND * 100.0f / 2.0f) - CALIBRATION_OFFSET_CM) : 0.0f, 1);
  Serial.print(",exceso="); Serial.print(best_excess, 4);
  Serial.print(",segundo="); Serial.print(second_best_score, 4);
  Serial.print(",candidato="); Serial.println(candidate_detected ? 1 : 0);
  appendMeasurement(distance_cm, best_score, score_threshold, detected,
                    best_index, best_excess, second_best_score,
                    candidate_detected);

  char lcd_line_0[17];
  char lcd_line_1[17];
  lcdClear();

  if (detected) {
    Serial.print("Blanco en: "); Serial.print(distance_cm, 1);
    Serial.print(" cm | Score: "); Serial.println(best_score, 3);
    Serial.print("Pico secundario: "); Serial.print(second_best_score, 3);
    Serial.print(" | Prominente: "); Serial.println(prominent_peak ? "SI" : "NO");
    snprintf(lcd_line_0, sizeof(lcd_line_0), "D:%5.1f cm", distance_cm);
    snprintf(lcd_line_1, sizeof(lcd_line_1), "S:%.3f T:%.3f", best_score, score_threshold);
  } else {
    Serial.print("Sin blancos. Max: "); Serial.print(best_score, 3);
    Serial.print(" | Umbral: "); Serial.println(score_threshold, 3);
    snprintf(lcd_line_0, sizeof(lcd_line_0), "Sin objeto");
    snprintf(lcd_line_1, sizeof(lcd_line_1), "S:%.3f T:%.3f", best_score, score_threshold);
  }

  lcdSetCursor(0, 0);
  lcdPrint(lcd_line_0);
  lcdSetCursor(0, 1);
  lcdPrint(lcd_line_1);
  delay(RESULT_HOLD_MS);
}