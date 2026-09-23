#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ==========================================
// 1. PINES Y HARDWARE
// ==========================================
#define TONE_PIN 4        // Salida PWM hacia el amplificador TPA3110 (Chirp acústico)
#define MIC_ADC_PIN 1     // Entrada analógica conectada a la salida del micrófono KY-038

#define I2C_SDA_PIN 8     // Bus de datos I2C para pantalla LCD
#define I2C_SCL_PIN 9     // Bus de reloj I2C para pantalla LCD
#define LCD_ADDR 0x3F     // Dirección hexadecimal del expansor PCF8574

// ==========================================
// 2. PARÁMETROS DEL RADAR (24kHz)
// ==========================================
// Fundamento LFM (Linear Frequency Modulation):
// Se emite un barrido continuo de frecuencia que varía linealmente en el tiempo.
// k_rate (tasa de chirp) = (F_STOP - F_START) / T_chirp = 6000 Hz / 0.05 s = 120,000 Hz/s
#define F_START 2000                 // Frecuencia inicial del barrido: 2 kHz
#define F_STOP  8000                 // Frecuencia final del barrido: 8 kHz
#define CHIRP_DURATION_US 50000ULL   // Duración temporal del pulso de transmisión T = 50 ms (50,000 us)
#define NUM_CHIRP_STEPS 100          // Discretización del barrido: 100 escalones de frecuencia

// Criterio de Nyquist-Shannon: Para muestrear 8 kHz sin aliasing se requiere Fs > 16 kHz.
// Con Fs = 24 kHz, la frecuencia máxima representable teórica es 12 kHz (margen de 4 kHz).
#define FS 24000ULL                  // Frecuencia de muestreo: 24,000 muestras por segundo (Ts = 41.66 us)
#define TX_SAMPLES 1200              // Muestras del pulso transmitido: N_tx = Fs * T = 24000 * 0.05 = 1200 muestras
#define RX_SAMPLES 2400              // Ventana de escucha: N_rx = 2400 muestras -> Tiempo total = 100 ms

// Tamaño de la FFT: Potencia de 2 para algoritmo radix-2.
// Para convolución lineal sin aliasing temporal circular se requiere:
// FFT_SIZE >= N_rx + N_tx - 1 = 2400 + 1200 - 1 = 3599. Se selecciona la potencia inmediata: 4096.
#define FFT_SIZE 4096                

#define SPEED_OF_SOUND 343.42f       // Velocidad del sonido en el aire a ~20 °C (m/s)
#define CALIBRATION_OFFSET_CM 2.9f   // Compensación de retraso instrumental (retardo en filtros y propagación mecánica)

// Delimitación de ventana temporal (Time-of-Flight):
// Distancia d = (k * c) / (2 * Fs)
// Índice 50  -> t = 50 / 24000 = 2.083 ms -> d_ida_vuelta = 0.715 m -> d_blanco = 35.7 cm brutos (~32.8 cm netos)
// Índice 200 -> t = 200 / 24000 = 8.333 ms -> d_ida_vuelta = 2.861 m -> d_blanco = 143.0 cm brutos (~140.1 cm netos)
#define MIN_SEARCH_INDEX 50          // Límite inferior: descarta acoplamiento acústico directo y ringing inicial
#define MAX_SEARCH_INDEX 200         // Límite superior: restringe la búsqueda a un alcance máximo de ~1.4 metros
#define ENERGY_THRESHOLD 53000.0f    // Umbral de magnitud de correlación para discriminar señal sobre ruido térmico

// ==========================================
// 3. ESTRUCTURAS DE MEMORIA Y SINCRONIZACIÓN
// ==========================================
uint16_t *rx_buffer;       // Búfer para muestras directas del ADC (12 bits: 0 a 4095)

// Vectores en frecuencia y tiempo para procesamiento complejo:
// X: Señal de referencia (pulso transmitido precalculado)
// Y: Señal capturada por el micrófono
float *real_X, *imag_X;    
float *real_Y, *imag_Y;    
float *bg_envelope;        // Perfil promedio de energía del entorno estacionario (ruido de fondo)

// Banderas de control atómico entre núcleos (inter-core signaling sin latencia de RTOS):
volatile bool prepare_chirp = false;
volatile bool chirp_ready = false;
volatile bool fire_chirp = false;

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
// 5. MOTOR FFT (Transformada Rápida de Fourier)
// ==========================================
// Implementación del algoritmo Cooley-Tukey en base 2 por Diezmado en el Tiempo (DIT).
// Complejidad algorítmica: O(N * log2(N)) vs O(N^2) de la DFT estándar.
void swapFloat(float &a, float &b) {
  float temp = a; a = b; b = temp;
}

void computeFFT(float* real, float* imag, uint16_t n, bool inverse) {
  // 1. Reordenamiento por inversión de bits (Bit-Reversal Permutation)
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

  // 2. Cálculo por etapas de mariposas (Butterfly Computations)
  for (uint16_t step = 1; step < n; step <<= 1) {
    // Factores de rotación (Twiddle Factors): W_N^k = e^(±j * pi / step)
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

        // Multiplicación compleja: t = u * data[idx2]
        float t_real = u_real * real[idx2] - u_imag * imag[idx2];
        float t_imag = u_real * imag[idx2] + u_imag * real[idx2];

        // Operación de mariposa: combinación de suma y resta
        real[idx2] = real[idx1] - t_real;
        imag[idx2] = imag[idx1] - t_imag;
        real[idx1] += t_real;
        imag[idx1] += t_imag;

        // Actualización iterativa del factor de rotación u = u * W
        float next_u_real = u_real * w_real - u_imag * w_imag;
        u_imag = u_real * w_imag + u_imag * w_real;
        u_real = next_u_real;
      }
    }
  }

  // Factor de normalización 1/N requerido para la IFFT (Transformada Inversa)
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
  // Asignación de arreglos en PSRAM externa para liberar la memoria interna SRAM
  rx_buffer = (uint16_t*) heap_caps_malloc(RX_SAMPLES * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
  real_X = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_X = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  real_Y = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  imag_Y = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  bg_envelope = (float*) heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);

  for (int i = 0; i < FFT_SIZE; i++) bg_envelope[i] = 0.0f;
}

// Núcleo 0 (PRO_CPU): Tarea dedicada exclusivamente a modular el hardware de transmisión
void chirpTaskCode(void * pvParameters) {
  uint32_t step_delay_us = CHIRP_DURATION_US / NUM_CHIRP_STEPS; // 500 us por escalón
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS; // 60 Hz por escalón
  
  while(true) {
    if (prepare_chirp) {
      prepare_chirp = false;
      chirp_ready = true; 
      
      // Bucle de espera activa: asegura latencia cero al recibir el disparo
      while(!fire_chirp) { } 
      fire_chirp = false;
      
      int64_t start_t = esp_timer_get_time();
      for(int i = 0; i < NUM_CHIRP_STEPS; i++) {
        // Generación escalonada aproximada del chirp LFM
        ledcWriteTone(TONE_PIN, F_START + (uint32_t)(i * freq_step));
        int64_t target = start_t + (i + 1) * step_delay_us;
        while(esp_timer_get_time() < target); // Temporización determinista en hardware
      }
      ledcWriteTone(TONE_PIN, 0); // Apagar emisión tras completar los 50 ms
    }
    vTaskDelay(1); 
  }
}

// Núcleo 1 (APP_CPU): Adquisición síncrona determinista del convertidor analógico-digital
void acquireSignal() {
  float sample_interval_us = 1000000.0f / FS; // Intervalo de muestreo Ts = 41.666 us
  
  prepare_chirp = true;
  while(!chirp_ready) { delay(1); } 
  chirp_ready = false;
  
  fire_chirp = true; // Señal simultánea de disparo al transmisor
  
  int64_t start_time = esp_timer_get_time();
  float next_sample_time = (float)start_time;

  // Bucle de muestreo estricto: evita jitter temporal en el ADC
  for (int i = 0; i < RX_SAMPLES; i++) {
    int64_t current_time = esp_timer_get_time();
    while (current_time < (int64_t)next_sample_time) {
      current_time = esp_timer_get_time();
    }
    next_sample_time += sample_interval_us;
    
    rx_buffer[i] = analogRead(MIC_ADC_PIN);
  }
}

// Síntesis analítica del pulso LFM de referencia y precomputación de su espectro conjugado:
// La señal continua modelada es: x(t) = sin(2*pi*(f0*t + 0.5*k*t^2))
// La correlación cruzada r_xy(t) equivale a la convolución x(-t) * y(t).
// En el dominio de Fourier: R_xy(f) = Y(f) * X*(f), donde X*(f) es el complejo conjugado de X(f).
void precalculateChirpFFT() {
  for (int i = 0; i < FFT_SIZE; i++) { real_X[i] = 0.0f; imag_X[i] = 0.0f; }
  
  float t_total = (float)CHIRP_DURATION_US / 1000000.0f; // 0.05 s
  float k_rate = (float)(F_STOP - F_START) / t_total;     // Tasa de cambio de frecuencia (Hz/s)
  
  // Generación temporal de la señal x[n]
  for (int n = 0; n < TX_SAMPLES; n++) {
    float t = (float)n / (float)FS;
    float phase = 2.0f * PI * (F_START * t + 0.5f * k_rate * t * t); // Fase cuadrática instantánea
    real_X[n] = sin(phase);
  }
  
  // Transformada hacia el dominio de la frecuencia: X(f) = FFT{x[n]}
  computeFFT(real_X, imag_X, FFT_SIZE, false);
  
  // Conjugación compleja directa: X*(f) = Real{X(f)} - j*Imag{X(f)}
  // Se almacena invertida para multiplicar directamente en el cálculo de correlación
  for (int i = 0; i < FFT_SIZE; i++) imag_X[i] = -imag_X[i];
}

// Filtro Adaptado (Matched Filter) y Sustracción de Envolvente Estática
void computeCrossCorrelation(bool is_calibration, int cal_count = 1) {
  // 1. Remoción del nivel de continua (Offset DC):
  // El ADC oscila entre 0 y 4095 centrado en ~2048. Restar la media centra la señal en 0V.
  float dc_sum = 0;
  for (int i = 0; i < RX_SAMPLES; i++) dc_sum += rx_buffer[i];
  float dc_offset = dc_sum / RX_SAMPLES;

  // 2. Cero-Padding (Zero-Padding) hasta completar 4096 muestras:
  // Extiende la señal para evitar el traslape circular de la FFT periódica
  for (int i = 0; i < FFT_SIZE; i++) {
    if (i < RX_SAMPLES) real_Y[i] = (float)rx_buffer[i] - dc_offset;
    else real_Y[i] = 0.0f;
    imag_Y[i] = 0.0f;
  }

  // 3. Transformada de Fourier de la señal recibida: Y(f) = FFT{y[n]}
  computeFFT(real_Y, imag_Y, FFT_SIZE, false);

  // 4. Multiplicación espectral: R_xy(f) = Y(f) * X*(f)
  // Producto de dos números complejos (a + jb) * (c + jd):
  // Parte Real = a*c - b*d, Parte Imaginaria = a*d + b*c
  for (int i = 0; i < FFT_SIZE; i++) {
    float r = real_Y[i] * real_X[i] - imag_Y[i] * imag_X[i];
    float im = real_Y[i] * imag_X[i] + imag_Y[i] * real_X[i];
    real_Y[i] = r;
    imag_Y[i] = im;
  }

  // 5. Transformada Inversa: r_xy[k] = IFFT{R_xy(f)}
  // Convierte el espectro de vuelta al dominio temporal (retardo k)
  computeFFT(real_Y, imag_Y, FFT_SIZE, true);

  // 6. Extracción de Envolvente y Cancelación de Ruido Estático:
  // Se obtiene el módulo de la señal analítica (|real_Y[i]|).
  // La resta diferencial elimina reflexiones invariables (paredes, mesa, rebote mutuo parlante-micrófono).
  for (int i = 0; i < FFT_SIZE; i++) {
    float current_envelope = abs(real_Y[i]);
    if (is_calibration) {
      // Promedio acumulado de fondo durante la etapa de calibración
      bg_envelope[i] += current_envelope / (float)cal_count;
    } else {
      // Señal neta = Envolvente actual - Mapa estático de fondo
      real_Y[i] = current_envelope - bg_envelope[i];
      if (real_Y[i] < 0) real_Y[i] = 0; // Rectificación de media onda (anula residuos negativos)
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  ledcAttach(TONE_PIN, F_START, 8);
  ledcWriteTone(TONE_PIN, 0);

  lcdInit();
  lcdSetCursor(0, 0); lcdPrint("Iniciando PSRAM");

  allocateMemory();
  precalculateChirpFFT();

  // Despliegue de la tarea generadora del pulso en el Núcleo 0
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
  
  // Rutina de línea base: 5 pasadas para estimar el vector espectral estacionario bg_envelope
  int cal_passes = 5;
  for (int c = 0; c < cal_passes; c++) {
    Serial.print("Muestra de calibracion "); Serial.print(c + 1); Serial.println("/5");
    acquireSignal();
    computeCrossCorrelation(true, cal_passes); 
    delay(1000); 
  }
  
  lcdClear();
  lcdSetCursor(0, 0); lcdPrint("Radar Listo!");
  Serial.println("Calibracion profunda completa y promediada.");
}

void loop() {
  delay(2000);

  acquireSignal();
  computeCrossCorrelation(false);

  int best_index = -1;
  float max_energy = 0.0f;

  // Algoritmo de Detección de Pico:
  // Se evalúa la envolvente correlacionada únicamente dentro del rango de interés [50, 200).
  for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
    if (real_Y[k] > max_energy) {
      max_energy = real_Y[k];
      best_index = k; // Índice k representa el retraso en cantidad de muestras (delta_n)
    }
  }

  char lcd_buf[16];
  lcdSetCursor(0, 1);

  // Criterio de Decisión de Blanco:
  // Requiere un índice temporal válido y que la energía neta supere el umbral preestablecido
  if (best_index > 0 && max_energy > ENERGY_THRESHOLD) {
    // 1. Cálculo del Tiempo de Vuelo (Time of Flight - ToF):
    // t_retardo (s) = k_muestras / Frecuencia_de_muestreo (muestras/s)
    float time_delay_s = (float)best_index / (float)FS;
    
    // 2. Modelo Físico de Propagación Bidireccional (Eco):
    // Distancia total recorrida = t_retardo * c
    // Como el sonido viaja ida y vuelta hacia el objeto:
    // d_unidireccional = (t_retardo * c) / 2
    // d_cm = d_unidireccional * 100
    // d_medida = d_cm - CALIBRATION_OFFSET_CM
    float distance_cm = (((time_delay_s * SPEED_OF_SOUND) / 2.0f) * 100.0f) - CALIBRATION_OFFSET_CM;

    Serial.print("Blanco en: "); Serial.print(distance_cm, 1);
    Serial.print(" cm | Energia: "); Serial.println(max_energy, 1);
    
    snprintf(lcd_buf, sizeof(lcd_buf), "Dist: %4.1f cm  ", distance_cm);
  } else {
    // Si la energía máxima no supera el umbral, se asume ausencia de blanco reflectivo
    Serial.print("Sin blancos. Energia fondo residual: "); Serial.println(max_energy, 1);
    snprintf(lcd_buf, sizeof(lcd_buf), "Buscando...     ");
  }
  
  lcdPrint(lcd_buf);
}