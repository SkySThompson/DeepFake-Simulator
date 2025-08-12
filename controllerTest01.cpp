// experiment_controller.cpp
#include "content_database.h"
#include "camera_biometrics.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

struct BiometricReading {
    double heartRate;
    double stressLevel;
    long timestamp;
    bool isHuman;  // Ground truth
    int contentId;
};

class ExperimentController {
private:
    CameraBiometrics detector;
    ContentDatabase contentDB;
    std::vector<BiometricReading> readings;
    std::ofstream dataFile;
    
public:
    void runExperiment() {
        // Initialize camera
        if(!detector.initialize()) {
            std::cout << "Failed to initialize camera biometrics!" << std::endl;
            return;
        }
        
        // Open data file
        dataFile.open("biometric_data.csv");
        dataFile << "timestamp,heart_rate,stress_level,is_human,content_id" << std::endl;
        
        auto samples = contentDB.getShuffledSamples();
        
        std::cout << "=== BIOMETRIC AI DETECTOR EXPERIMENT ===" << std::endl;
        std::cout << "Setup:" << std::endl;
        std::cout << "1. Turn on phone flashlight" << std::endl;
        std::cout << "2. Place finger on camera + flash (gentle pressure)" << std::endl;
        std::cout << "3. Keep finger there throughout entire experiment" << std::endl;
        std::cout << "4. Read each text sample carefully" << std::endl;
        std::cout << "5. Press ENTER after reading each one" << std::endl << std::endl;
        std::cout << "Ready? Press ENTER to begin..." << std::endl;
        std::cin.get();
        
        std::cout << "Recording biometric data continuously. Press ENTER to stop." << std::endl;
        
        auto samples = contentDB.getShuffledSamples();
        
        for (size_t i = 0; i < samples.size(); ++i) {
            auto& sample = samples[i];
            std::cout << "\n=== Sample " << (i+1) << "/" << samples.size() << " ===" << std::endl;
            std::cout << "Keep finger on camera. Read this text:" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            std::cout << sample.text << std::endl << std::endl;
            std::cout << "Reading... (Press ENTER when finished)" << std::endl;
            std::cin.get();
            
            std::cout << "\nRecording your biometric response..." << std::endl;
            collectBiometrics(sample, 5000);  // Record for 5 seconds after reading
            
            std::cout << "✓ Data collected." << std::endl;
        }
        
        dataFile.close();
        std::cout << "\n🎉 Experiment complete! Data saved to biometric_data.csv" << std::endl;
        std::cout << "Collected " << readings.size() << " biometric readings." << std::endl;
    }
    
private:
    void collectBiometrics(ContentDatabase::ContentSample sample, int durationMs) {
        auto endTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(durationMs);
        
        while(std::chrono::steady_clock::now() < endTime) {
            auto reading = detector.captureReading();
            
            BiometricReading dataPoint;
            dataPoint.heartRate = reading.heartRate;
            dataPoint.stressLevel = reading.stressLevel;
            dataPoint.timestamp = reading.timestamp;
            dataPoint.isHuman = sample.isHuman;
            dataPoint.contentId = sample.sampleId;
            
            readings.push_back(dataPoint);
            
            // Save to CSV
            dataFile << dataPoint.timestamp << "," << dataPoint.heartRate << "," 
                    << dataPoint.stressLevel << "," << dataPoint.isHuman << "," 
                    << dataPoint.contentId << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30fps
        }
        // Keep the video window open for an additional 2 seconds after data collection
        std::cout << "Holding video display for 2 more seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
};