#include "Logger.h"

Logger::Logger() {}

void Logger::write(const String& data) {
    if (data.length() > 0) {
        if (data == "S") {
            Serial.println(data);
        } else {
            Serial.println("[EVT] " + data);
        }
    }

    Serial.flush();
}

void Logger::write(int data) {
    String output = "[ENC] " + String(data);
    Serial.println(output);

    Serial.flush();
}

void Logger::write(float data) {
    String output = "[ENC] " + String(data, 2);
    Serial.println(output);

    Serial.flush();
}

void Logger::writeRaw(uint16_t data) {
    String output = "[RAW] " + String(data);
    Serial.println(output);
    Serial.flush();
}

String Logger::read() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        if (line.length() > 0) {
            return line;
        }
    }
    return "";
}

void Logger::ack() {
    Serial.println("A");
    Serial.flush();
}
