// ============================================================================
//  CARACTERIZACION DEL MICROFONO/PREAMPLIFICADOR - ESP32-S3
//
//  OBJETIVO: medir COMO ES REALMENTE tu hardware (KY-038 + transistor BC549)
//  antes de seguir asumiendo numeros de simulacion. Este sketch NO calcula
//  distancias, NO emite ningun chirp ni usa el tweeter/TPA3110 para nada:
//  SOLO ESCUCHA. Es puramente pasivo, sobre la sensibilidad cruda del mic
//  y el preamp con sonidos que TU generas con la mano/voz (aplausos,
//  chasquidos, golpes, voz). Captura ventanas de audio y guarda sus
//  estadisticas (y, cada cierto numero de eventos, la forma de onda cruda)
//  en un log persistente en flash, igual que el firmware del radar.
//
//  QUE GUARDA POR CADA EVENTO (linea "E," en el log):
//    marca, minimo, maximo, pico-a-pico, DC (offset), RMS (sobre DC),
//    y si hubo saturacion (clipping) contra 0 o 4095.
//  Cada PROFILE_EVERY eventos guarda ademas la forma de onda cruda completa
//  (linea "W,") para que pueda ver el chirp/eco tal cual llega al ADC.
//
//  DOS MODOS DE CAPTURA (cambia CAPTURE_MODE):
//    1 = TRIGGER (recomendado, activo por defecto): monitorea continuamente
//                    y captura la ventana en cuanto el nivel supera un
//                    umbral (chasquido, aplauso, voz, golpe). Si pasan
//                    TRIGGER_IDLE_MS sin ningun sonido fuerte, captura igual
//                    (asi silencio_total/ambiente_normal tambien dejan datos).
//    0 = MANUAL   : cada 'g' en el monitor serial (o cada 2 s solo) captura
//                    una ventana de 50 ms. Sirve si prefieres sincronizar
//                    tu por teclado en vez de que dispare solo.
//
//  MARCAS: igual que el radar, MARKS_CM avanza con el boton BOOT (GPIO0) y
//  gueda escrita en cada linea, para que sepamos que bloque de prueba es cada
//  dato cuando yo revise el log.
//
//  COMANDOS SERIAL (115200): d=volcar log | X=borrar log | s=estado |
//                              n=siguiente marca | g=capturar ahora | h=ayuda
// ============================================================================
#include <Arduino.h>
#include <math.h>
#include <FS.h>
#include <LittleFS.h>
#include <Preferences.h>

// ==========================================
// 1. PINES Y CONFIGURACION
// ==========================================
#define MIC_ADC_PIN   1
#define BTN_PIN       0
#define SERIAL_BAUD   115200

#define CAPTURE_MODE  1            // 0 = manual (serial 'g' o automatico), 1 = por umbral (trigger, recomendado)

#define FS            24000ULL    // mismo muestreo que el radar
#define WIN_SAMPLES   1200        // 50 ms por evento (igual a la duracion del chirp)
#define RAW_SAVE_N    400         // cuantas muestras crudas guardar por evento con perfil (~16.7 ms)
#define PROFILE_EVERY 3           // cada cuantos eventos se guarda la forma de onda cruda

#define AUTO_PERIOD_MS  2000      // en modo manual, si no escribes 'g', captura solo cada X ms
#define TRIGGER_LEVEL   80        // modo trigger: disparo si |muestra-2048| supera esto
#define TRIGGER_ARM_MS  400       // tiempo minimo entre disparos (evita doble-disparo del mismo golpe)
#define TRIGGER_IDLE_MS 3000      // si no hubo trigger en este tiempo, captura igual (para silencio_total)

#define LOG_PATH      "/mic_test_log.csv"
#define BTN_DEBOUNCE_MS 250
#define BOOT_WINDOW_S   6

// Marcas de prueba. TODAS son sonidos que TU generas (voz, dedos, golpes).
// No hay chirp ni tweeter involucrado: es puramente sensibilidad del mic/preamp.
// Edita libremente y avisame que pusiste en cada una si cambias algo.
static const char* MARKS[] = {
  "silencio_total",        // no hables, no te muevas, nada prendido cerca si es posible
  "ambiente_normal",       // el cuarto como esta normalmente (PC, ventilador, etc.), sin hablar
  "chasquido_10cm",        // chasquido de dedos a 10 cm del mic
  "chasquido_30cm",        // chasquido de dedos a 30 cm del mic
  "chasquido_50cm",        // chasquido de dedos a 50 cm del mic
  "aplauso_30cm",          // aplauso franco a 30 cm del mic
  "aplauso_100cm",         // aplauso franco a 1 m del mic
  "voz_baja_30cm",         // hablar bajito a 30 cm
  "voz_normal_30cm",       // hablar normal a 30 cm
  "golpe_mesa_fuerte",     // golpear la mesa con fuerza (buscando saturacion)
  "silbido_agudo_30cm"     // silbido o tono agudo sostenido a 30 cm (banda alta del mic)
};
#define NUM_MARKS ((int)(sizeof(MARKS)/sizeof(MARKS[0])))

// ==========================================
// 2. ESTADO
// ==========================================
uint16_t rx_buf[WIN_SAMPLES];
Preferences prefs;
bool     fs_ok=false, log_full=false;
size_t   log_limit=0;
uint32_t boot_id=0;
int      mark_idx=0, ev_in_mark=0;
uint32_t ev_count=0;
uint32_t last_auto_ms=0, last_trigger_ms=0;
volatile bool btn_flag=false;
volatile uint32_t btn_last=0;
static char logbuf[3200];

// ==========================================
// 3. CAPTURA DETERMINISTA (enteros de 64 bits, sin drift)
// ==========================================
void captureWindow(){
  int64_t start=esp_timer_get_time();
  for(int i=0;i<WIN_SAMPLES;i++){
    int64_t target=start+((int64_t)i*1000000LL)/(int64_t)FS;
    int64_t now=esp_timer_get_time();
    while(now<target) now=esp_timer_get_time();
    rx_buf[i]=analogRead(MIC_ADC_PIN);
  }
}

// ==========================================
// 4. LOG PERSISTENTE (mismo esquema que el radar)
// ==========================================
bool logWrite(const char* s,size_t n){
  if(!fs_ok||log_full) return false;
  File f=LittleFS.open(LOG_PATH,FILE_APPEND);
  if(!f) return false;
  size_t w=f.write((const uint8_t*)s,n);
  size_t sz=f.size();
  f.close();
  if(sz>log_limit) log_full=true;
  return w==n;
}
void logBootHeader(){
  int n=snprintf(logbuf,sizeof(logbuf),
    "#BOOT,boot=%lu,fs=%llu,win_samples=%d,mode=%d,trigger_level=%d,marks=%d\n",
    (unsigned long)boot_id,(unsigned long long)FS,WIN_SAMPLES,CAPTURE_MODE,TRIGGER_LEVEL,NUM_MARKS);
  logWrite(logbuf,n);
  static const char cols[]=
    "#COLS E: E,boot,ms,mark_idx,mark_name,n_in_mark,min,max,pp,dc,rms,clip_lo,clip_hi\n"
    "#COLS W: W,boot,ms,mark_idx,mark_name,n_in_mark,raw[0..RAW_SAVE_N-1]\n";
  logWrite(cols,strlen(cols));
}
void logMark(){
  int n=snprintf(logbuf,sizeof(logbuf),"#MARK,boot=%lu,ms=%lu,mark_idx=%d,mark_name=%s\n",
                 (unsigned long)boot_id,(unsigned long)millis(),mark_idx,MARKS[mark_idx]);
  logWrite(logbuf,n);
}
void dumpLog(){
  if(!fs_ok){ Serial.println("FS no disponible (revisa la particion con SPIFFS/LittleFS)"); return; }
  File f=LittleFS.open(LOG_PATH,FILE_READ);
  if(!f){ Serial.println("(log vacio)"); return; }
  Serial.println("=====INICIO_LOG=====");
  uint8_t buf[256];
  while(f.available()){ size_t n=f.read(buf,sizeof(buf)); if(!n) break; Serial.write(buf,n); }
  f.close();
  Serial.println("\n=====FIN_LOG=====");
}
void clearLog(){
  if(fs_ok) LittleFS.remove(LOG_PATH);
  log_full=false;
  Serial.println("Log borrado.");
  logBootHeader();
}
void printStatus(){
  Serial.printf("boot=%lu marca=%d(%s) eventos=%lu fs_ok=%d ",
    (unsigned long)boot_id,mark_idx,MARKS[mark_idx],(unsigned long)ev_count,fs_ok);
  if(fs_ok) Serial.printf("log=%u/%u bytes full=%d",(unsigned)LittleFS.usedBytes(),(unsigned)log_limit,log_full);
  Serial.println();
}

// ==========================================
// 5. ANALISIS Y REGISTRO DE UN EVENTO
// ==========================================
void processAndLog(){
  int mn=4095, mx=0; double sum=0;
  for(int i=0;i<WIN_SAMPLES;i++){ int v=rx_buf[i]; if(v<mn)mn=v; if(v>mx)mx=v; sum+=v; }
  double dc=sum/WIN_SAMPLES;
  double acc=0; for(int i=0;i<WIN_SAMPLES;i++){ double d=(double)rx_buf[i]-dc; acc+=d*d; }
  double rms=sqrt(acc/WIN_SAMPLES);
  int clip_lo=0, clip_hi=0;
  for(int i=0;i<WIN_SAMPLES;i++){ if(rx_buf[i]<=1) clip_lo++; if(rx_buf[i]>=4094) clip_hi++; }

  ev_in_mark++; ev_count++;
  int n=snprintf(logbuf,sizeof(logbuf),
    "E,%lu,%lu,%d,%s,%d,%d,%d,%d,%.1f,%.1f,%d,%d\n",
    (unsigned long)boot_id,(unsigned long)millis(),mark_idx,MARKS[mark_idx],ev_in_mark,
    mn,mx,mx-mn,dc,rms,clip_lo,clip_hi);
  logWrite(logbuf,n);

  Serial.printf("[%s #%d] min=%d max=%d pp=%d dc=%.0f rms=%.1f clip(lo,hi)=(%d,%d)\n",
    MARKS[mark_idx],ev_in_mark,mn,mx,mx-mn,dc,rms,clip_lo,clip_hi);

  if(ev_count%PROFILE_EVERY==0){
    n=snprintf(logbuf,sizeof(logbuf),"W,%lu,%lu,%d,%s,%d",
               (unsigned long)boot_id,(unsigned long)millis(),mark_idx,MARKS[mark_idx],ev_in_mark);
    int lim=(RAW_SAVE_N<WIN_SAMPLES)?RAW_SAVE_N:WIN_SAMPLES;
    for(int i=0;i<lim && n<(int)sizeof(logbuf)-8;i++) n+=snprintf(logbuf+n,sizeof(logbuf)-n,",%u",rx_buf[i]);
    logbuf[n++]='\n';
    logWrite(logbuf,n);
    Serial.println("  (forma de onda cruda guardada en el log)");
  }
}

// ==========================================
// 6. BOTON, MARCAS, COMANDOS
// ==========================================
void IRAM_ATTR onBtn(){
  uint32_t now=millis();
  if(now-btn_last>BTN_DEBOUNCE_MS){ btn_last=now; btn_flag=true; }
}
void nextMark(){
  mark_idx=(mark_idx+1)%NUM_MARKS; ev_in_mark=0;
  logMark();
  Serial.printf("\n>>> MARCA -> [%d] %s <<<\n",mark_idx,MARKS[mark_idx]);
}
void printHelp(){
  Serial.println("Comandos: g=capturar ahora | d=volcar log | X=borrar log | s=estado | n=siguiente marca | h=ayuda");
  Serial.print("Marcas disponibles: ");
  for(int i=0;i<NUM_MARKS;i++) Serial.printf("[%d]%s ",i,MARKS[i]);
  Serial.println();
}
void handleSerial(){
  while(Serial.available()){
    char ch=(char)Serial.read();
    if(ch=='d'||ch=='D') dumpLog();
    else if(ch=='x'||ch=='X') clearLog();
    else if(ch=='s'||ch=='S') printStatus();
    else if(ch=='n'||ch=='N') nextMark();
    else if(ch=='g'||ch=='G'){ captureWindow(); processAndLog(); }
    else if(ch=='h'||ch=='H'||ch=='?') printHelp();
  }
}

// ==========================================
// 7. SETUP / LOOP
// ==========================================
void setup(){
  Serial.begin(SERIAL_BAUD);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(BTN_PIN,INPUT_PULLUP);
  attachInterrupt(BTN_PIN,onBtn,FALLING);

  fs_ok=LittleFS.begin(true);
  if(fs_ok){ log_limit=(size_t)((uint64_t)LittleFS.totalBytes()*85/100); if(log_limit>1000000) log_limit=1000000; }
  prefs.begin("mictest",false);
  boot_id=prefs.getUInt("boot",0)+1; prefs.putUInt("boot",boot_id);

  Serial.printf("\n== TEST DE MICROFONO boot %lu | modo=%s | log %s ==\n",
    (unsigned long)boot_id, CAPTURE_MODE==0?"MANUAL":"TRIGGER", fs_ok?"OK":"NO DISPONIBLE");
  printHelp();
  Serial.println("Tienes 6 s para pedir 'd' (log anterior) antes de iniciar...");
  for(int s=BOOT_WINDOW_S;s>0;s--){
    for(int i=0;i<50;i++){ handleSerial(); delay(20); }
  }

  logBootHeader();
  mark_idx=0; ev_in_mark=0; logMark();
  Serial.printf("\nListo. Marca actual: [0] %s\n",MARKS[0]);
  Serial.println("Pulsa BOOT (o manda 'n') para pasar a la siguiente marca cuando termines cada bloque.");
  last_auto_ms=millis();
}

void loop(){
  handleSerial();
  if(btn_flag){ btn_flag=false; nextMark(); }

#if CAPTURE_MODE == 0
  // Modo manual: 'g' captura al instante; si no se manda nada, captura sola cada AUTO_PERIOD_MS
  if(millis()-last_auto_ms>=AUTO_PERIOD_MS){
    last_auto_ms=millis();
    captureWindow();
    processAndLog();
  }
  delay(20);
#else
  // Modo trigger: mira continuamente 1 muestra a la vez; si cruza el umbral, captura la ventana completa.
  // Si pasa TRIGGER_IDLE_MS sin ningun disparo, captura igual (asi "silencio_total" / "ambiente_normal"
  // tambien dejan datos, aunque nunca crucen el umbral).
  int v=analogRead(MIC_ADC_PIN);
  bool by_level = abs(v-2048)>TRIGGER_LEVEL && millis()-last_trigger_ms>TRIGGER_ARM_MS;
  bool by_idle  = millis()-last_trigger_ms>TRIGGER_IDLE_MS;
  if(by_level || by_idle){
    last_trigger_ms=millis();
    captureWindow();       // ya perdimos ~1 muestra de margen, es aceptable para este proposito
    processAndLog();
  }
#endif
}
