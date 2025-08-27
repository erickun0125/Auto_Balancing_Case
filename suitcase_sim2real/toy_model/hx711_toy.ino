/* 이중 HX711 로드셀 - 간단 버전 */
#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

// 두 HX711 핀 설정
const int HX711_1_dout = 4;
const int HX711_1_sck = 5;
const int HX711_2_dout = 6;
const int HX711_2_sck = 7;

// HX711 객체
HX711_ADC LoadCell1(HX711_1_dout, HX711_1_sck);
HX711_ADC LoadCell2(HX711_2_dout, HX711_2_sck);

const int calVal_eepromAdress = 0;
unsigned long lastSensorRead = 0;
const int SENSOR_INTERVAL = 10; // 100Hz

const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
  Serial.begin(115200);
  delay(10);
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  
  LoadCell1.begin();
  LoadCell2.begin();
  
  LoadCell1.setSamplesInUse(1);
  LoadCell2.setSamplesInUse(1);
  
  unsigned long stabilizingtime = 2000;
  boolean _tare = true;
  
  LoadCell1.start(stabilizingtime, _tare);
  LoadCell2.start(stabilizingtime, _tare);
  
  if (LoadCell1.getTareTimeoutFlag() || LoadCell1.getSignalTimeoutFlag() ||
      LoadCell2.getTareTimeoutFlag() || LoadCell2.getSignalTimeoutFlag()) {
    Serial.println("ERROR:HX711_TIMEOUT");
    while (1);
  }
  
  // EEPROM 캘리브레이션 로드
  float cal1, cal2;
  EEPROM.get(0, cal1);
  EEPROM.get(0, cal2);
  if (isnan(cal1)) cal1 = 1.0;
  if (isnan(cal2)) cal2 = 1.0;
  
  LoadCell1.setCalFactor(cal1);
  LoadCell2.setCalFactor(cal2);
  
  Serial.println("READY");
  
  while (!LoadCell1.update() || !LoadCell2.update());
}

void loop() {
  static boolean newDataReady = false;
  unsigned long currentTime = millis();
  
  if (LoadCell1.update()) newDataReady = true;
  if (LoadCell2.update()) newDataReady = true;
  
  if (newDataReady && (currentTime - lastSensorRead >= SENSOR_INTERVAL)) {
    
    float val1 = LoadCell1.getData();
    float val2 = LoadCell2.getData();
    
    // 평균을 빼고 큰 값을 2배
    float avg = (val1 + val2) / 2.0;
    float diff1 = val1 - avg;
    float diff2 = val2 - avg;
    
    
    // CSV 전송: timestamp,processed_value,button,tare_status

    Serial.print(diff1, 3);
    Serial.print(",");
    Serial.print(diff2, 3);
    Serial.print(",");
    Serial.print(digitalRead(BUTTON_PIN));
    Serial.print(",");
    Serial.print((LoadCell1.getTareStatus() || LoadCell2.getTareStatus()) ? 1 : 0);
    Serial.println();
    
    newDataReady = false;
    lastSensorRead = currentTime;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  
  // 명령 처리
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "TARE") {
      LoadCell1.tareNoDelay();
      LoadCell2.tareNoDelay();
      Serial.println("TARE_STARTED");
    }
    else if (cmd == "STATUS") {
      Serial.println("STATUS:OK");
    }
  }
  
  if (LoadCell1.getTareStatus() || LoadCell2.getTareStatus()) {
    Serial.println("TARE_COMPLETE");
  }
}