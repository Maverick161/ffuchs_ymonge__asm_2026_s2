#include <Arduino.h>

// --- Pines ---
#define TONE_PIN 23        // Salida PWM hacia TPA3110
#define MIC_ADC_PIN 35     // Entrada analógica AO del KY-038 (ADC1)

// --- Parámetros del Radar ---
#define F_START 2000       // 2 kHz
#define F_STOP  6000       // 6 kHz
#define CHIRP_DURATION_MS 20
#define NUM_CHIRP_STEPS 50

#define BUFFER_SIZE 1024       // 1024 muestras = ~51.2 ms
#define SAMPLE_INTERVAL_US 50  // 50 us -> Fs = 20 kHz

uint16_t rx_buffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);

  // Configuración ADC en GPIO35
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_ADC_PIN, ADC_6db);

  // Configuración PWM Tx (Core 3.x)
  ledcAttach(TONE_PIN, F_START, 8);
  ledcWrite(TONE_PIN, 0);

  delay(1000);
  Serial.println("Iniciando Test 03: Loopback Tx/Rx en GPIO35");
}

void loop() {
  uint32_t step_delay_us = (CHIRP_DURATION_MS * 1000) / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;
  
  uint32_t chirp_step_idx = 0;
  uint32_t next_chirp_update = micros();
  uint32_t next_sample_time = micros();

  // 1. Iniciar emisión de chirp
  ledcWriteTone(TONE_PIN, F_START);

  // 2. Transmisión y muestreo sincronizado
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while ((int32_t)(micros() - next_sample_time) < 0) {
      // Sincronización precisa Fs = 20 kHz
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

  // Apagar emisión al completar la ventana
  ledcWriteTone(TONE_PIN, 0);

  // 3. Volcar buffer al Serial Plotter
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.print("Min:1500,Max:4000,Captura:");
    Serial.println(rx_buffer[i]);
  }

  delay(1500); // Pausa entre disparos
}