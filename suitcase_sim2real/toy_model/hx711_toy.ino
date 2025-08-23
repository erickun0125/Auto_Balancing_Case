/*
   로드셀 센서 데이터를 Python으로 실시간 스트리밍
   HX711 ADC 라이브러리 사용
   수정: 기존 디버깅 코드 → 실시간 데이터 스트림
*/

#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

//pins:
const int HX711_dout = 4; //mcu > HX711 dout pin
const int HX711_sck = 5; //mcu > HX711 sck pin

//HX711 constructor:
HX711_ADC LoadCell(HX711_dout, HX711_sck);

const int calVal_eepromAdress = 0;
unsigned long lastSensorRead = 0;
const int SENSOR_INTERVAL = 10; // 10ms = 100Hz

// 추가 센서들 (예시)
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
  Serial.begin(115200); // Python 호환을 위해 115200으로 변경
  delay(10);
  
  // 버튼과 LED 설정
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  
  LoadCell.begin();
  
  // 실시간 성능을 위한 설정
  LoadCell.setSamplesInUse(1); // 최고 응답성을 위해 1로 설정
  
  unsigned long stabilizingtime = 2000;
  boolean _tare = true;
  LoadCell.start(stabilizingtime, _tare);
  
  if (LoadCell.getTareTimeoutFlag() || LoadCell.getSignalTimeoutFlag()) {
    // 에러 발생 시 특별한 메시지 전송
    Serial.println("ERROR:HX711_TIMEOUT");
    while (1);
  }
  else {
    // EEPROM에서 calibration 값 로드 시도
    float calibrationValue;
    EEPROM.get(calVal_eepromAdress, calibrationValue);
    if (isnan(calibrationValue)) {
      calibrationValue = 1.0; // 기본값
    }
    LoadCell.setCalFactor(calibrationValue);
    
    // 초기화 완료 신호
    Serial.println("READY");
  }
  
  while (!LoadCell.update());
}

void loop() {
  static boolean newDataReady = 0;
  unsigned long currentTime = millis();
  
  // 센서 데이터 업데이트
  if (LoadCell.update()) newDataReady = true;
  
  // 정해진 주기마다 데이터 전송
  if (newDataReady && (currentTime - lastSensorRead >= SENSOR_INTERVAL)) {
    
    // 모든 센서 값 읽기
    float loadCellValue = LoadCell.getData();
    int buttonState = digitalRead(BUTTON_PIN);
    unsigned long timestamp = millis();
    
    // CSV 형태로 데이터 전송 (Python에서 파싱하기 쉬움)
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(loadCellValue, 3); // 소수점 3자리
    Serial.print(",");
    Serial.print(buttonState);
    Serial.print(",");
    Serial.print(LoadCell.getTareStatus() ? 1 : 0); // Tare 상태
    Serial.println(); // 줄바꿈으로 패킷 구분
    
    newDataReady = 0;
    lastSensorRead = currentTime;
    
    // LED 깜빡임으로 동작 확인
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  
  // 시리얼 명령 처리 (Python에서 제어 가능)
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "TARE") {
      LoadCell.tareNoDelay();
      Serial.println("TARE_STARTED");
    }
    else if (command == "STATUS") {
      Serial.print("STATUS:");
      Serial.print("CAL=");
      Serial.print(LoadCell.getCalFactor(), 6);
      Serial.print(",SAMPLES=");
      Serial.print(LoadCell.getSamplesInUse());
      Serial.println();
    }
    else if (command.startsWith("CAL=")) {
      float newCal = command.substring(4).toFloat();
      if (newCal != 0) {
        LoadCell.setCalFactor(newCal);
        Serial.print("CAL_SET:");
        Serial.println(newCal, 6);
      }
    }
    else if (command.startsWith("SAMPLES=")) {
      int newSamples = command.substring(8).toInt();
      if (newSamples > 0 && newSamples <= 16) {
        LoadCell.setSamplesInUse(newSamples);
        Serial.print("SAMPLES_SET:");
        Serial.println(newSamples);
      }
    }
  }
  
  // Tare 완료 확인
  if (LoadCell.getTareStatus() == true) {
    Serial.println("TARE_COMPLETE");
  }
}