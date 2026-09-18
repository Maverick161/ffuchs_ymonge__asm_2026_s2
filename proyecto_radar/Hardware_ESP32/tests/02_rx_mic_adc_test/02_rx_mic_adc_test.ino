#include <Arduino.h>

// GPIO1 pertenece al ADC1 (ADC1_CH0), seguro y sin conflictos con el Wi-Fi
#define MIC_ADC_PIN 1

void setup() {
  Serial.begin(115200);
  
  // Resolución de 12 bits (0 a 4095)
  analogReadResolution(12);
  
  // Atenuación de 11dB para permitir lecturas hasta ~3.1V (cubre perfecto tus 1.67V pico)
  analogSetAttenuation(ADC_11db); 
}

void loop() {
  int raw_val = analogRead(MIC_ADC_PIN);

  // Formato para fijar los límites visuales en el Serial Plotter de Arduino
  Serial.print("Min:0,Max:4095,Señal:");
  Serial.println(raw_val);
  
  // Pequeño retardo para no saturar el puerto serie en esta prueba visual
  delay(2);
}