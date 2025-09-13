/*
 * HX711 Load Cell Interface for Auto Balancing Case - 작동하는 버전
 * HX711_ADC 라이브러리 사용
 */

#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

// 6개 HX711 핀 설정 (기존 프로젝트 핀 맵 유지)
const int DOUT_FR = 2, SCK_FR = 3;
const int DOUT_RR = 4, SCK_RR = 5;
const int DOUT_FL = 6, SCK_FL = 7;
const int DOUT_RL = 8, SCK_RL = 9;
const int DOUT_HANDLE_1 = 10, SCK_HANDLE_1 = 11;
const int DOUT_HANDLE_2 = 12, SCK_HANDLE_2 = 13;

// HX711_ADC 객체들
HX711_ADC scale_wheel_FR(DOUT_FR, SCK_FR);
HX711_ADC scale_wheel_RR(DOUT_RR, SCK_RR);
HX711_ADC scale_wheel_FL(DOUT_FL, SCK_FL);
HX711_ADC scale_wheel_RL(DOUT_RL, SCK_RL);
HX711_ADC scale_handle_1(DOUT_HANDLE_1, SCK_HANDLE_1);
HX711_ADC scale_handle_2(DOUT_HANDLE_2, SCK_HANDLE_2);

const int SEND_INTERVAL = 100; // 10Hz (100ms)
unsigned long lastSendTime = 0;

void setup() {
  Serial.begin(115200);
  delay(100);
  
  // 모든 로드셀 초기화
  scale_wheel_FR.begin();
  scale_wheel_RR.begin();
  scale_wheel_FL.begin();
  scale_wheel_RL.begin();
  scale_handle_1.begin();
  scale_handle_2.begin();
  
  // 샘플 수 설정
  scale_wheel_FR.setSamplesInUse(1);
  scale_wheel_RR.setSamplesInUse(1);
  scale_wheel_FL.setSamplesInUse(1);
  scale_wheel_RL.setSamplesInUse(1);
  scale_handle_1.setSamplesInUse(1);
  scale_handle_2.setSamplesInUse(1);
  
  // 안정화 시간 및 Tare
  unsigned long stabilizingtime = 2000;
  boolean _tare = true;
  
  scale_wheel_FR.start(stabilizingtime, _tare);
  scale_wheel_RR.start(stabilizingtime, _tare);
  scale_wheel_FL.start(stabilizingtime, _tare);
  scale_wheel_RL.start(stabilizingtime, _tare);
  scale_handle_1.start(stabilizingtime, _tare);
  scale_handle_2.start(stabilizingtime, _tare);
  
  // 타임아웃 체크
  if (scale_wheel_FR.getTareTimeoutFlag() || scale_wheel_RR.getTareTimeoutFlag() ||
      scale_wheel_FL.getTareTimeoutFlag() || scale_wheel_RL.getTareTimeoutFlag() ||
      scale_handle_1.getTareTimeoutFlag() || scale_handle_2.getTareTimeoutFlag()) {
    Serial.println("ERROR:HX711_TIMEOUT");
    while (1);
  }
  
}

void loop() {
  static boolean newDataReady = false;
  
  // 데이터 업데이트 체크
  if (scale_wheel_FR.update()) newDataReady = true;
  if (scale_wheel_RR.update()) newDataReady = true;
  if (scale_wheel_FL.update()) newDataReady = true;
  if (scale_wheel_RL.update()) newDataReady = true;
  if (scale_handle_1.update()) newDataReady = true;
  if (scale_handle_2.update()) newDataReady = true;
  
  // 데이터 전송 (10Hz)
  if (newDataReady && (millis() - lastSendTime >= SEND_INTERVAL)) {
    sendSensorData();
    newDataReady = false;
    lastSendTime = millis();
  }
  
  // 명령 처리
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }
}

void sendSensorData() {
  // CSV 형식으로 전송: FR,RR,FL,RL,H1,H2
  Serial.print(scale_wheel_FR.getData(), 2);
  Serial.print(",");
  Serial.print(scale_wheel_RR.getData(), 2);
  Serial.print(",");
  Serial.print(scale_wheel_FL.getData(), 2);
  Serial.print(",");
  Serial.print(scale_wheel_RL.getData(), 2);
  Serial.print(",");
  Serial.print(scale_handle_1.getData(), 2);
  Serial.print(",");
  Serial.println(scale_handle_2.getData(), 2);
}

void handleCommand(String command) {
  if (command == "TARE_ALL") {
    scale_wheel_FR.tareNoDelay();
    scale_wheel_RR.tareNoDelay();
    scale_wheel_FL.tareNoDelay();
    scale_wheel_RL.tareNoDelay();
    scale_handle_1.tareNoDelay();
    scale_handle_2.tareNoDelay();
    Serial.println("TARE_STARTED");
  }
  else if (command == "STATUS") {
    Serial.println("STATUS:OK");
  }
  else {
    Serial.println("UNKNOWN_COMMAND:" + command);
  }
}