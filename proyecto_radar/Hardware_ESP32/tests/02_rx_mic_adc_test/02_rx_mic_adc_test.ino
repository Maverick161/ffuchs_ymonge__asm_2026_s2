#include <Arduino.h>

#define MIC_ADC_PIN 34

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_ADC_PIN, ADC_6db); // Escala lineal de 0 a ~2.2V
}

void loop() {
  int raw_val = analogRead(MIC_ADC_PIN);

  // Escala visible en Serial Plotter
  Serial.print("Min:0,Max:4095,Señal:");
  Serial.println(raw_val);
  delay(2);
}