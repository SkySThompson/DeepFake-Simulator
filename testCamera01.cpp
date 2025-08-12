// test_camera.cpp
#include "camera_biometrics.h"
#include <iostream>

int main() {
    CameraBiometrics detector;
    
    if (!detector.initialize()) {
        std::cout << "Failed to initialize camera!" << std::endl;
        return -1;
    }
    
    std::cout << "=== CAMERA BIOMETRIC TEST ===" << std::endl;
    std::cout << "1. Turn on your phone's flashlight" << std::endl;
    std::cout << "2. Place finger gently over camera lens + flash" << std::endl;
    std::cout << "3. Hold steady for 10 seconds" << std::endl;
    std::cout << "Press any key to start..." << std::endl;
    std::cin.get();
    
    for (int i = 0; i < 300; i++) {  // 10 seconds at 30fps
        auto reading = detector.captureReading();
        
        if (i % 30 == 0) {  // Print every second
            std::cout << "HR: " << reading.heartRate << " BPM, "
                     << "Stress: " << (reading.stressLevel * 100) << "%" << std::endl;
        }
    }
    
    std::cout << "Test complete!" << std::endl;
    return 0;
}