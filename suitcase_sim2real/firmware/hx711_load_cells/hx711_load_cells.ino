/*
 * HX711 Load Cell Interface for Auto Balancing Case
 * Reads 6 HX711 ADC modules and outputs CSV at 10 Hz via serial.
 * Uses the HX711_ADC library.
 */

#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

// 6 HX711 pin assignments (matching hardware wiring)
const int DOUT_FR = 2, SCK_FR = 3;
const int DOUT_RR = 4, SCK_RR = 5;
const int DOUT_FL = 6, SCK_FL = 7;
const int DOUT_RL = 8, SCK_RL = 9;
const int DOUT_HANDLE_1 = 10, SCK_HANDLE_1 = 11;
const int DOUT_HANDLE_2 = 12, SCK_HANDLE_2 = 13;

// HX711_ADC objects (4 wheel + 2 handle channels)
HX711_ADC scale_wheel_FR(DOUT_FR, SCK_FR);
HX711_ADC scale_wheel_RR(DOUT_RR, SCK_RR);
HX711_ADC scale_wheel_FL(DOUT_FL, SCK_FL);
HX711_ADC scale_wheel_RL(DOUT_RL, SCK_RL);
HX711_ADC scale_handle_1(DOUT_HANDLE_1, SCK_HANDLE_1);
HX711_ADC scale_handle_2(DOUT_HANDLE_2, SCK_HANDLE_2);

const int SEND_INTERVAL = 100; // 10 Hz (100 ms)
unsigned long lastSendTime = 0;

void setup() {
  Serial.begin(115200);
  delay(100);

  // Initialize all load cells
  scale_wheel_FR.begin();
  scale_wheel_RR.begin();
  scale_wheel_FL.begin();
  scale_wheel_RL.begin();
  scale_handle_1.begin();
  scale_handle_2.begin();

  // Set samples per reading (1 = fastest, no averaging)
  scale_wheel_FR.setSamplesInUse(1);
  scale_wheel_RR.setSamplesInUse(1);
  scale_wheel_FL.setSamplesInUse(1);
  scale_wheel_RL.setSamplesInUse(1);
  scale_handle_1.setSamplesInUse(1);
  scale_handle_2.setSamplesInUse(1);

  // Stabilization delay and initial tare
  unsigned long stabilizingtime = 2000;
  boolean _tare = true;

  scale_wheel_FR.start(stabilizingtime, _tare);
  scale_wheel_RR.start(stabilizingtime, _tare);
  scale_wheel_FL.start(stabilizingtime, _tare);
  scale_wheel_RL.start(stabilizingtime, _tare);
  scale_handle_1.start(stabilizingtime, _tare);
  scale_handle_2.start(stabilizingtime, _tare);

  // Check for tare timeout errors
  if (scale_wheel_FR.getTareTimeoutFlag() || scale_wheel_RR.getTareTimeoutFlag() ||
      scale_wheel_FL.getTareTimeoutFlag() || scale_wheel_RL.getTareTimeoutFlag() ||
      scale_handle_1.getTareTimeoutFlag() || scale_handle_2.getTareTimeoutFlag()) {
    Serial.println("ERROR:HX711_TIMEOUT");
    while (1);
  }

}

void loop() {
  static boolean newDataReady = false;

  // Poll for new ADC data
  if (scale_wheel_FR.update()) newDataReady = true;
  if (scale_wheel_RR.update()) newDataReady = true;
  if (scale_wheel_FL.update()) newDataReady = true;
  if (scale_wheel_RL.update()) newDataReady = true;
  if (scale_handle_1.update()) newDataReady = true;
  if (scale_handle_2.update()) newDataReady = true;

  // Send data at 10 Hz
  if (newDataReady && (millis() - lastSendTime >= SEND_INTERVAL)) {
    sendSensorData();
    newDataReady = false;
    lastSendTime = millis();
  }

  // Process incoming serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }
}

void sendSensorData() {
  // CSV format: FR,RR,FL,RL,H1,H2
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
