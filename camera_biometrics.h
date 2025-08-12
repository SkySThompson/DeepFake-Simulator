// camera_biometrics.h
#ifndef CAMERA_BIOMETRICS_H
#define CAMERA_BIOMETRICS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <numeric>

class CameraBiometrics {
private:
    cv::VideoCapture cap;
    std::vector<double> greenValues;
    std::vector<std::chrono::steady_clock::time_point> timestamps;
    bool flashOn = false;
    
public:
    struct BiometricReading {
        double heartRate;
        double stressLevel;  // Based on heart rate variability
        long timestamp;
    };
    
    bool initialize(int deviceIndex = 0) {
        cap.open(deviceIndex);  // Open specified camera device
        if (!cap.isOpened()) {
            std::cout << "Error: Cannot open camera device " << deviceIndex << "!" << std::endl;
            return false;
        }
        
        // Set camera properties for better heart rate detection
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        return true;
    }
    
    BiometricReading captureReading() {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            return {0, 0, 0};
        }
        
        // Extract green channel (blood absorption changes most in green light)
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        cv::Mat greenChannel = channels[1];
        
        // Calculate average green intensity in center region (fingertip area)
        cv::Rect roi(frame.cols/2 - 50, frame.rows/2 - 50, 100, 100);
        cv::Mat centerRegion = greenChannel(roi);
        
        double avgGreen = cv::mean(centerRegion)[0];
        
        // Store reading with timestamp
        auto now = std::chrono::steady_clock::now();
        greenValues.push_back(avgGreen);
        timestamps.push_back(now);
        
        // Keep only last 300 readings (10 seconds at 30fps)
        if (greenValues.size() > 300) {
            greenValues.erase(greenValues.begin());
            timestamps.erase(timestamps.begin());
        }
        
        // Calculate heart rate using FFT approximation
        double heartRate = calculateHeartRate();
        double stressLevel = calculateStressLevel();
        
        // Display feedback
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 3);
        
        // Use larger font scale and thickness for better readability
        double fontScale = 1.2;
        int thickness = 3;
        int baseline = 0;
        
        // Background rectangle for text for better contrast
        int padding = 10;
        cv::Size textSize = cv::getTextSize("Place finger over camera + flash", cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(frame, cv::Point(20, 20), cv::Point(20 + textSize.width + padding, 20 + textSize.height + padding), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, "Place finger over camera + flash", 
                   cv::Point(25, 20 + textSize.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
        
        std::string hrText = "Heart Rate: " + std::to_string((int)heartRate) + " BPM";
        textSize = cv::getTextSize(hrText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(frame, cv::Point(20, 60), cv::Point(20 + textSize.width + padding, 60 + textSize.height + padding), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, hrText, 
                   cv::Point(25, 60 + textSize.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), thickness);
        
        std::string stressText = "Stress Level: " + std::to_string((int)(stressLevel * 100)) + "%";
        textSize = cv::getTextSize(stressText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(frame, cv::Point(20, 100), cv::Point(20 + textSize.width + padding, 100 + textSize.height + padding), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stressText, 
                   cv::Point(25, 100 + textSize.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thickness);
        
        cv::imshow("Biometric Monitor", frame);
        cv::waitKey(1);
        
        return {
            heartRate,
            stressLevel,
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
        };
    }
    
private:
    double calculateHeartRate() {
        if (greenValues.size() < 60) return 0;  // Need at least 2 seconds of data
        
        // Improved peak detection with threshold and minimum peak distance
        std::vector<double> filtered = smoothSignal(greenValues);
        std::vector<int> peaks;
        double meanVal = std::accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
        int minPeakDistance = 15; // minimum samples between peaks (~0.5s at 30fps)
        
        for (int i = 1; i < filtered.size() - 1; i++) {
            if (filtered[i] > filtered[i-1] && filtered[i] > filtered[i+1] && filtered[i] > meanVal) {
                if (peaks.empty() || (i - peaks.back()) > minPeakDistance) {
                    peaks.push_back(i);
                }
            }
        }
        
        if (peaks.size() < 2) return 0;
        
        // Calculate average time between peaks
        double totalTime = 0;
        for (int i = 1; i < peaks.size(); i++) {
            auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamps[peaks[i]] - timestamps[peaks[i-1]]);
            totalTime += timeDiff.count();
        }
        
        double avgPeriodMs = totalTime / (peaks.size() - 1);
        double heartRate = 60000.0 / avgPeriodMs;  // Convert to BPM
        
        // Clamp to realistic range
        return std::max(50.0, std::min(150.0, heartRate));
    }
    
    double calculateStressLevel() {
        if (greenValues.size() < 30) return 0;
        
        // Heart rate variability as stress indicator
        double mean = std::accumulate(greenValues.end()-30, greenValues.end(), 0.0) / 30.0;
        double variance = 0;
        
        for (auto it = greenValues.end()-30; it != greenValues.end(); ++it) {
            variance += (*it - mean) * (*it - mean);
        }
        variance /= 29.0;
        
        // Higher variability = more stress (simplified)
        return std::min(1.0, variance / 1000.0);
    }
    
    std::vector<double> smoothSignal(const std::vector<double>& signal) {
        std::vector<double> smoothed;
        int windowSize = 5;
        
        for (int i = 0; i < signal.size(); i++) {
            double sum = 0;
            int count = 0;
            
            for (int j = std::max(0, i - windowSize/2); 
                 j <= std::min((int)signal.size()-1, i + windowSize/2); j++) {
                sum += signal[j];
                count++;
            }
            
            smoothed.push_back(sum / count);
        }
        
        return smoothed;
    }
    
    std::vector<int> findPeaks(const std::vector<double>& signal) {
        std::vector<int> peaks;
        
        for (int i = 1; i < signal.size() - 1; i++) {
            if (signal[i] > signal[i-1] && signal[i] > signal[i+1] && 
                signal[i] > (std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size())) {
                peaks.push_back(i);
            }
        }
        
        return peaks;
    }
};

#endif