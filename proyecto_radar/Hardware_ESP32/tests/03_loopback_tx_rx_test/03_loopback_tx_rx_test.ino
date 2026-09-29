#include <Arduino.h>

// --- Pines ESP32-S3 ---
#define TONE_PIN 4         // Salida PWM hacia filtro/TPA3110
#define MIC_ADC_PIN 1      // Entrada analógica del preamplificador (ADC1_CH0)

// --- Parámetros del Radar (Alineado con el simulador Python) ---
#define F_START 2000       // Frecuencia inicial: 2 kHz
#define F_STOP  8000       // Frecuencia final: 8 kHz
#define CHIRP_DURATION_MS 50 // Duración del pulso: 50 ms
#define NUM_CHIRP_STEPS 100  // Aumentado a 100 para un barrido más suave

#define SAMPLE_RATE 48000  // Frecuencia de muestreo estricta a 48 kHz
#define BUFFER_SIZE 2400   // 2400 muestras exactas en 50 ms a 48 kHz

uint16_t rx_buffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);

  // Configuración ADC (Core 3.x)
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Permite lecturas de 0V a ~3.1V

  // Configuración PWM Tx (Core 3.x)
  ledcAttach(TONE_PIN, F_START, 8);
  ledcWriteTone(TONE_PIN, 0); // Inicia silenciado

  delay(1000);
  Serial.println("Iniciando Test: Loopback Tx/Rx (Chirp 2k-8k, 48kHz) en ESP32-S3");
}

void loop() {
  uint32_t step_delay_us = (CHIRP_DURATION_MS * 1000) / NUM_CHIRP_STEPS;
  float freq_step = (float)(F_STOP - F_START) / NUM_CHIRP_STEPS;
  
  uint32_t chirp_step_idx = 0;
  
  // Usamos float para evitar pérdida de precisión con el periodo de 20.833 microsegundos
  float sample_interval_us = 1000000.0 / SAMPLE_RATE; 
  
  uint32_t start_time = micros();
  uint32_t next_chirp_update = start_time;
  float next_sample_time = start_time;

  // 1. Iniciar emisión de chirp
  ledcWriteTone(TONE_PIN, F_START);

  // 2. Transmisión y muestreo sincronizado (Ventana de 50 ms)
  for (int i = 0; i < BUFFER_SIZE; i++) {
    
    // Espera activa (polling) muy precisa para los 48 kHz
    while (micros() < (uint32_t)next_sample_time) {
      // Bloqueo hasta que toque la siguiente muestra
    }
    next_sample_time += sample_interval_us;

    // Captura inmediata para reducir el jitter
    rx_buffer[i] = analogRead(MIC_ADC_PIN);

    // Actualización de la frecuencia del chirp (Tx)
    if (chirp_step_idx < NUM_CHIRP_STEPS && micros() >= next_chirp_update) {
      chirp_step_idx++;
      if (chirp_step_idx < NUM_CHIRP_STEPS) {
        uint32_t f = F_START + (uint32_t)(chirp_step_idx * freq_step);
        ledcWriteTone(TONE_PIN, f);
        next_chirp_update += step_delay_us;
      } else {
        ledcWriteTone(TONE_PIN, 0); // Silencio preventivo al finalizar los pasos
      }
    }
  }

  // Apagar emisión al completar la ventana de 2400 muestras
  ledcWriteTone(TONE_PIN, 0);

  // 3. Volcar buffer al Serial Plotter
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Escala fija para ver el comportamiento del punto de reposo (~1000 a 1600)
    Serial.print("Min:0,Max:4095,Captura:");
    Serial.println(rx_buffer[i]);
  }

  delay(1500); // Pausa entre disparos para visualizar la gráfica en el Plotter
}