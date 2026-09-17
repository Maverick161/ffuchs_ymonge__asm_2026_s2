#include <Arduino.h>

// --- Pines de Hardware ---
#define TONE_PIN 23        // Salida PWM hacia el TPA3110
#define MIC_ADC_PIN 35     // Entrada analógica KY-038 (ADC1)

// --- Parámetros de Transmisión (Chirp Ultracorto) ---
#define F_START 3000
#define F_STOP  7000
#define CHIRP_DURATION_US 1500       // 1.5 ms de duración
#define NUM_CHIRP_STEPS 20

// --- Parámetros de Muestreo (Fs = 25 kHz) ---
#define FS 25000
#define SAMPLE_INTERVAL_US (1000000 / FS) // 40 us por muestra
#define RX_BUFFER_SIZE 800                // 32 ms de captura en RAM
#define TX_SAMPLES ((CHIRP_DURATION_US * FS) / 1000000) // ~37 muestras

#define SPEED_OF_SOUND_M_S 343.0f
#define CALIBRATION_OFFSET_CM 4.5f       // Resta el retardo constante (34.7 - 4.7 cm)

// --- Umbrales de Detección ---
#define MIN_SEARCH_INDEX 50               // Ignora todo acoplamiento por debajo de ~28 cm
#define ENERGY_THRESHOLD 135000           // Rebaja levemente el umbral para facilitar detección a 40 cm
#define MAX_SEARCH_INDEX 130              // Límite superior (~85 cm)

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
  analogSetPinAttenuation(MIC_ADC_PIN, ADC_6db);

  ledcAttach(TONE_PIN, F_START, 8);
  ledcWrite(TONE_PIN, 0);

  generarChirpReferencia();
  delay(1000);
  Serial.println("=== Radar Acústico Calibrado (Test 04 Final) ===");
}

void loop() {
  uint32_t step_delay_us = CHIRP_DURATION_US / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;

  uint32_t chirp_step_idx = 0;
  uint32_t next_chirp_update = micros();
  uint32_t next_sample_time = micros();

  // 1. Transmisión del Chirp
  ledcWriteTone(TONE_PIN, F_START);

  // 2. Muestreo sincronizado a 25 kHz
  for (int i = 0; i < RX_BUFFER_SIZE; i++) {
    while ((int32_t)(micros() - next_sample_time) < 0) {
      // Espera activa precisa
    }
    next_sample_time += SAMPLE_INTERVAL_US;

    rx_buffer[i] = analogRead(MIC_ADC_PIN);

    if (chirp_step_idx < NUM_CHIRP_STEPS && (int32_t)(micros() - next_chirp_update) >= 0) {
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

  // 4. Correlación Cruzada (Matched Filter)
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

  // 5. Cálculo y filtrado de distancia
  if (best_index > 0 && max_energy > ENERGY_THRESHOLD) {
    float time_delay_s = (float)best_index / (float)FS;
    float distance_raw_cm = ((time_delay_s * SPEED_OF_SOUND_M_S) / 2.0f) * 100.0f;
    float distance_cm = distance_raw_cm - CALIBRATION_OFFSET_CM;

    Serial.print("Blanco -> Distancia: ");
    Serial.print(distance_cm, 1);
    Serial.print(" cm | Pico: ");
    Serial.println((long)max_energy);
  } else {
    Serial.print("Sin obstáculo en rango | Base: ");
    Serial.println((long)max_energy);
  }

  delay(1000);
}