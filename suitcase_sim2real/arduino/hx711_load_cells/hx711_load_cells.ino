/*
 * HX711 Load Cell Interface for Auto Balancing Case
 * Arduino code for reading 5 HX711 load cells and sending data via serial
 * 
 * Hardware connections:
 * - wheel_FR: DOUT=2, SCK=3
 * - wheel_RR: DOUT=4, SCK=5  
 * - wheel_FL: DOUT=6, SCK=7
 * - wheel_RL: DOUT=8, SCK=9
 * - handle:   DOUT=10, SCK=11
 */

#include "HX711.h"

// HX711 instances for each load cell
HX711 scale_wheel_FR;
HX711 scale_wheel_RR;  
HX711 scale_wheel_FL;
HX711 scale_wheel_RL;
HX711 scale_handle;

// Pin definitions
const int DOUT_FR = 2, SCK_FR = 3;
const int DOUT_RR = 4, SCK_RR = 5;
const int DOUT_FL = 6, SCK_FL = 7;
const int DOUT_RL = 8, SCK_RL = 9;
const int DOUT_HANDLE = 10, SCK_HANDLE = 11;

// Calibration factors (adjust these based on your load cells)
float calibration_factor_FR = 1.0;
float calibration_factor_RR = 1.0;
float calibration_factor_FL = 1.0;
float calibration_factor_RL = 1.0;
float calibration_factor_handle = 1.0;

// Data reading frequency
const unsigned long READ_INTERVAL = 20; // 50Hz (20ms)
unsigned long lastReadTime = 0;

void setup() {
  Serial.begin(115200);
  
  // Initialize all HX711 instances
  scale_wheel_FR.begin(DOUT_FR, SCK_FR);
  scale_wheel_RR.begin(DOUT_RR, SCK_RR);
  scale_wheel_FL.begin(DOUT_FL, SCK_FL);
  scale_wheel_RL.begin(DOUT_RL, SCK_RL);
  scale_handle.begin(DOUT_HANDLE, SCK_HANDLE);
  
  // Set calibration factors
  scale_wheel_FR.set_scale(calibration_factor_FR);
  scale_wheel_RR.set_scale(calibration_factor_RR);
  scale_wheel_FL.set_scale(calibration_factor_FL);
  scale_wheel_RL.set_scale(calibration_factor_RL);
  scale_handle.set_scale(calibration_factor_handle);
  
  // Tare all scales
  tareAllScales();
  
  Serial.println("HX711 Load Cell Interface Ready");
  Serial.println("Commands: TARE_ALL, SET_CAL_FR <value>, SET_CAL_RR <value>, etc.");
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }
  
  // Read and send sensor data at specified frequency
  unsigned long currentTime = millis();
  if (currentTime - lastReadTime >= READ_INTERVAL) {
    sendSensorData();
    lastReadTime = currentTime;
  }
}

void sendSensorData() {
  // Read all load cells
  float force_FR = scale_wheel_FR.get_units(1);
  float force_RR = scale_wheel_RR.get_units(1);
  float force_FL = scale_wheel_FL.get_units(1);
  float force_RL = scale_wheel_RL.get_units(1);
  float force_handle = scale_handle.get_units(1);
  
  // Send as JSON format
  Serial.print("{\"wheel_FR\":");
  Serial.print(force_FR, 2);
  Serial.print(",\"wheel_RR\":");
  Serial.print(force_RR, 2);
  Serial.print(",\"wheel_FL\":");
  Serial.print(force_FL, 2);
  Serial.print(",\"wheel_RL\":");
  Serial.print(force_RL, 2);
  Serial.print(",\"handle\":");
  Serial.print(force_handle, 2);
  Serial.println("}");
}

void handleCommand(String command) {
  if (command == "TARE_ALL") {
    tareAllScales();
    Serial.println("All scales tared");
  }
  else if (command.startsWith("SET_CAL_FR ")) {
    float cal = command.substring(11).toFloat();
    calibration_factor_FR = cal;
    scale_wheel_FR.set_scale(cal);
    Serial.println("FR calibration factor set to " + String(cal));
  }
  else if (command.startsWith("SET_CAL_RR ")) {
    float cal = command.substring(11).toFloat();
    calibration_factor_RR = cal;
    scale_wheel_RR.set_scale(cal);
    Serial.println("RR calibration factor set to " + String(cal));
  }
  else if (command.startsWith("SET_CAL_FL ")) {
    float cal = command.substring(11).toFloat();
    calibration_factor_FL = cal;
    scale_wheel_FL.set_scale(cal);
    Serial.println("FL calibration factor set to " + String(cal));
  }
  else if (command.startsWith("SET_CAL_RL ")) {
    float cal = command.substring(11).toFloat();
    calibration_factor_RL = cal;
    scale_wheel_RL.set_scale(cal);
    Serial.println("RL calibration factor set to " + String(cal));
  }
  else if (command.startsWith("SET_CAL_HANDLE ")) {
    float cal = command.substring(15).toFloat();
    calibration_factor_handle = cal;
    scale_handle.set_scale(cal);
    Serial.println("Handle calibration factor set to " + String(cal));
  }
  else if (command == "GET_RAW") {
    // Send raw values for debugging
    Serial.print("Raw values - FR:");
    Serial.print(scale_wheel_FR.read());
    Serial.print(" RR:");
    Serial.print(scale_wheel_RR.read());
    Serial.print(" FL:");
    Serial.print(scale_wheel_FL.read());
    Serial.print(" RL:");
    Serial.print(scale_wheel_RL.read());
    Serial.print(" Handle:");
    Serial.println(scale_handle.read());
  }
  else {
    Serial.println("Unknown command: " + command);
  }
}

void tareAllScales() {
  scale_wheel_FR.tare();
  scale_wheel_RR.tare();
  scale_wheel_FL.tare();
  scale_wheel_RL.tare();
  scale_handle.tare();
}
