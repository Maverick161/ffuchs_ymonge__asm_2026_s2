#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- Pines I2C ESP32-S3 ---
#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9

// Configuración de la pantalla LCD I2C
// La dirección suele ser 0x27 (chip PCF8574T) o 0x3F (chip PCF8574AT)
#define LCD_ADDRESS 0x3F
#define LCD_COLUMNS 16
#define LCD_ROWS 2

LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);

void setup() {
  Serial.begin(115200);
  
  // Inicializar el bus I2C con los pines específicos del diseño
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  
  // Inicializar LCD
  lcd.init();
  lcd.backlight();
  
  // Render de prueba estático
  lcd.setCursor(0, 0);
  lcd.print("Radar Acustico");
  lcd.setCursor(0, 1);
  lcd.print("Iniciando...");
  
  Serial.println("Test 05: LCD inicializado. Buscando en 0x3F...");
  delay(2000);
}

void loop() {
  // Despliegue de variables simuladas
  float distancia_simulada = random(300, 1000) / 10.0; // 30.0 a 100.0 cm
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Blanco a:");
  
  lcd.setCursor(0, 1);
  lcd.print(distancia_simulada, 1);
  lcd.print(" cm");
  
  Serial.print("Render LCD -> ");
  Serial.println(distancia_simulada);
  
  delay(500); // Refresco visual cada medio segundo
}