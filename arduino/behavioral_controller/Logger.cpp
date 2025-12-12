#include "Logger.h"

Logger::Logger() {}

void Logger::write(const String& data) {
    if (data.length() == 0) {
        return;
    }

    if (data == "S") {
        Serial.println(F("S"));
    } else {
        Serial.print(F("[EVT] "));
        Serial.println(data);
    }

    Serial.flush();
}

void Logger::write(uint16_t data) {
    Serial.print(F("[RAW] "));
    Serial.println(data);

    Serial.flush();
}

void Logger::write(float data) {
    Serial.print(F("[ENC] "));
    Serial.println(data, 1);

    Serial.flush();
}

String Logger::read() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        if (line.length() > 0) {
            ack();
            return line;
        }
    }
    return "";
}

void Logger::ack() {
    Serial.println("A");
}
