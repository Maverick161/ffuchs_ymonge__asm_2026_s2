#include <Arduino.h>

// --- Configuración de Pines ESP32-S3 ---
#define TONE_PIN 4            // Salida de audio hacia el TPA3110
// #define MIC_PIN 1          // Pin para el ADC (lo usaremos más adelante)

// --- Parámetros del Chirp Acústico (Según modelo matemático) ---
#define F_START 2000          // Frecuencia inicial f0 = 2 kHz
#define F_STOP  8000          // Frecuencia final f1 = 8 kHz
#define CHIRP_DURATION_MS 50  // Duración T = 0.05 s (50 ms)
#define NUM_STEPS 100         // Pasos para suavizar el barrido PWM

void emitChirp() {
  // Calculamos cuánto tiempo dura cada paso de frecuencia
  uint32_t step_delay_us = (CHIRP_DURATION_MS * 1000) / NUM_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_STEPS;

  // Ejecutamos el barrido lineal de frecuencias
  for (int i = 0; i < NUM_STEPS; i++) {
    uint32_t current_freq = F_START + (uint32_t)(i * freq_step);
    ledcWriteTone(TONE_PIN, current_freq);
    delayMicroseconds(step_delay_us);
  }

  // Apagar la emisión al terminar el pulso (enmudecer)
  ledcWriteTone(TONE_PIN, 0);
}

void setup() {
  Serial.begin(115200);
  
  // API de Arduino Core v3.x para ESP32: ledcAttach(pin, freq, resolucion_bits)
  // Iniciamos en F_START con 8 bits de resolución PWM
  ledcAttach(TONE_PIN, F_START, 8);
  
  Serial.println("Transmisor listo: emitiendo chirp LFM de 2 kHz a 8 kHz...");
}

void loop() {
  emitChirp();
  
  // En tu simulación (experiment_echo_generation.py), la recepción asume 
  // una escucha donde el eco más lejano tarda ~312.5 ms (15000 muestras)
  // Damos un silencio de 500 ms (PRI) entre cada disparo para permitir el rebote.
  delay(500); 
}