#include <Arduino.h>
#include <math.h>

// --- Pines de Hardware ESP32-S3 ---
#define TONE_PIN 4
#define MIC_ADC_PIN 1

// --- Parámetros de Transmisión (Punto dulce físico) ---
#define F_START 2000
#define F_STOP  8000
#define CHIRP_DURATION_US 20000ULL   // 20 ms: El parlante reacciona y suena nítido
#define NUM_CHIRP_STEPS 60           // Barrido suave

// --- Parámetros de Muestreo ---
#define FS 48000ULL
#define TX_SAMPLES ((CHIRP_DURATION_US * FS) / 1000000ULL) // Exactamente 960 muestras
#define RX_BUFFER_SIZE 2400          // 50 ms totales de ventana

#define SPEED_OF_SOUND_M_S 343.42f
#define CALIBRATION_OFFSET_CM 17.2f

// --- Ventana de Búsqueda Estricta (30 cm a 1 metro) ---
// 60 muestras de retardo = ~1.66 ms = ~28 cm. Ignoramos el choque acústico directo.
#define MIN_SEARCH_INDEX 60          

// Elevamos el umbral para ignorar por completo el ruido de fondo
#define ENERGY_THRESHOLD 1500000     

// 335 muestras de retardo = ~7.0 ms = ~1.2 metros. Ignoramos el eco de las paredes.
#define MAX_SEARCH_INDEX 335

uint16_t rx_buffer[RX_BUFFER_SIZE];
int16_t tx_ref[TX_SAMPLES];

void generarChirpReferencia() {
  float t_total = (float)CHIRP_DURATION_US / 1000000.0f;
  float k_rate = (float)(F_STOP - F_START) / t_total;

  for (int n = 0; n < TX_SAMPLES; n++) {
    float t = (float)n / (float)FS;
    float phase = 2.0f * PI * (F_START * t + 0.5f * k_rate * t * t);
    tx_ref[n] = (int16_t)(sin(phase) * 127.0f);
  }
}

void setup() {
  Serial.begin(115200);
  
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  ledcAttach(TONE_PIN, F_START, 8);
  ledcWriteTone(TONE_PIN, 0);

  generarChirpReferencia();
  
  delay(1000);
  Serial.println("=== Radar Acústico: Correlación Directa (48 kHz) ===");
}

void loop() {
  uint32_t step_delay_us = CHIRP_DURATION_US / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;

  uint32_t chirp_step_idx = 0;
  float sample_interval_us = 1000000.0f / FS; 
  
  uint32_t start_time = micros();
  uint32_t next_chirp_update = start_time;
  float next_sample_time = start_time;

  // 1. Transmisión del Chirp
  ledcWriteTone(TONE_PIN, F_START);

  // 2. Muestreo de alta precisión a 48 kHz
  for (int i = 0; i < RX_BUFFER_SIZE; i++) {
    while (micros() < (uint32_t)next_sample_time) {
      // Polling de microsegundos
    }
    next_sample_time += sample_interval_us;
    rx_buffer[i] = analogRead(MIC_ADC_PIN);

    // Actualización simultánea del PWM
    if (chirp_step_idx < NUM_CHIRP_STEPS && micros() >= next_chirp_update) {
      chirp_step_idx++;
      if (chirp_step_idx < NUM_CHIRP_STEPS) {
        uint32_t f = F_START + (uint32_t)(chirp_step_idx * freq_step);
        ledcWriteTone(TONE_PIN, f);
        next_chirp_update += step_delay_us;
      } else {
        ledcWriteTone(TONE_PIN, 0);
      }
    }
  }
  ledcWriteTone(TONE_PIN, 0);

  // 3. Compensación de offset DC
  int32_t dc_sum = 0;
  for (int i = 0; i < RX_BUFFER_SIZE; i++) {
    dc_sum += rx_buffer[i];
  }
  int16_t dc_offset = dc_sum / RX_BUFFER_SIZE;

  // 4. Correlación Cruzada Directa
  int best_index = -1;
  int64_t max_energy = 0;

  for (int k = MIN_SEARCH_INDEX; k < MAX_SEARCH_INDEX; k++) {
    int64_t corr = 0;
    for (int n = 0; n < TX_SAMPLES; n++) {
      int16_t sample_ac = (int16_t)rx_buffer[k + n] - dc_offset;
      corr += (int64_t)sample_ac * tx_ref[n];
    }
    int64_t energy = abs(corr);
    if (energy > max_energy) {
      max_energy = energy;
      best_index = k;
    }
  }

  // 5. Cálculo monostático de distancia
  if (best_index > 0 && max_energy > ENERGY_THRESHOLD) {
    float time_delay_s = (float)best_index / (float)FS;
    float distance_raw_cm = ((time_delay_s * SPEED_OF_SOUND_M_S) / 2.0f) * 100.0f;
    float distance_cm = distance_raw_cm - CALIBRATION_OFFSET_CM;

    Serial.print("Eco detectado -> Distancia: ");
    Serial.print(distance_cm, 1);
    Serial.print(" cm | Pico correlación: ");
    Serial.println((long)max_energy);
  } else {
    Serial.print("Sin obstáculos | Energía máxima: ");
    Serial.println((long)max_energy);
  }

  delay(1000);
}