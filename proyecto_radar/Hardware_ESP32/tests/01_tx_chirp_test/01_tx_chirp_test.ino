#include <Arduino.h>

#define TONE_PIN 23
#define F_START 2000          // Frecuencia inicial: 2 kHz
#define F_STOP  6000          // Frecuencia final: 6 kHz
#define CHIRP_DURATION_MS 20  // Duración del pulso: 20 ms
#define NUM_STEPS 50          // Pasos discretos de frecuencia en el barrido

void emitChirp() {
  uint32_t step_delay_us = (CHIRP_DURATION_MS * 1000) / NUM_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_STEPS;

  for (int i = 0; i < NUM_STEPS; i++) {
    uint32_t current_freq = F_START + (uint32_t)(i * freq_step);
    ledcWriteTone(TONE_PIN, current_freq);
    delayMicroseconds(step_delay_us);
  }

  // Apagar la emisión al terminar el pulso (silencio entre disparos)
  ledcWriteTone(TONE_PIN, 0);
}

void setup() {
  Serial.begin(115200);
  ledcAttach(TONE_PIN, F_START, 8);
  Serial.println("Transmisor listo: emitiendo chirp LFM cada 500 ms...");
}

void loop() {
  emitChirp();
  delay(500); // Periodo de repetición de pulsos (PRI)
}