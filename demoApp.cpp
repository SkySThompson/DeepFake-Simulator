// demo_app.cpp
#include "camera01.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

class PersonalAIDetector {
private:
    CameraBiometrics detector;
    std::vector<double> thresholds;  // Learned from training data
    
public:
    struct PredictionResult {
        double confidence;
        bool predictedHuman;
        std::string reasoning;
        double heartRate;
        double stressLevel;
    };
    
    bool initialize() {
        return detector.initialize();
    }
    
    PredictionResult analyzeContent(std::string content) {
        std::cout << "=== ANALYZING CONTENT ===" << std::endl;
        std::cout << content << std::endl << std::endl;
        std::cout << "📱 Instructions:" << std::endl;
        std::cout << "1. Keep finger on camera + flash" << std::endl;
        std::cout << "2. Read the text above carefully" << std::endl;
        std::cout << "3. Your biometric response is being recorded..." << std::endl;
        std::cout << "4. Press ENTER when you've finished reading" << std::endl;
        
        // Collect baseline for 2 seconds before user reads# Biometric AI Content Detector - 3-Day Build Guide

## Project Overview
Build a personalized AI content detector using your biological responses (heart rate + skin conductance) to train a model that recognizes when YOU feel content is AI-generated vs human-made.

## Hardware Requirements (PHONE-BASED VERSION)
- **Smartphone with camera and flash** (you already have this!)
- **Laptop/desktop with webcam** (backup option)
- **C++ compiler** (GCC/Clang or Visual Studio)
- **OpenCV library** for camera access

## Future Hardware Upgrade (Post-Hackathon)
- Arduino Uno R3 ($25) + Pulse Sensor ($15) + GSR Sensor ($12)
- This would give you true GSR data and more accurate heart rate

---

## DAY 1: Camera-Based Biometric Setup

### Step 1: Install OpenCV (30 minutes)
```bash
# Linux/Mac
sudo apt-get install libopencv-dev
# OR brew install opencv

# Windows (using vcpkg)
vcpkg install opencv4[contrib,nonfree]:x64-windows