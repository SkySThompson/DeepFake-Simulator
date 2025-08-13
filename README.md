# DeepFake Simulator

A sophisticated C++ application for biometric analysis and deepfake detection using OpenCV and machine learning techniques.

## Overview

This project implements a comprehensive system for analyzing biometric data to detect deepfake content. It combines computer vision, machine learning, and biometric analysis to distinguish between authentic and synthetic media.

## Features

- **Real-time biometric monitoring**: Heart rate and stress level analysis
- **Content classification**: AI vs Human content detection
- **Camera integration**: Real-time video processing capabilities
- **Machine learning models**: Pre-trained detectors for various deepfake types
- **Data visualization**: Biometric data analysis and plotting

## Technical Stack

- **Language**: C++17
- **Computer Vision**: OpenCV
- **Machine Learning**: Custom ML models with Python integration
- **Data Processing**: Eigen library for linear algebra
- **Build System**: CMake (optional)

## Project Structure

```
├── main.cpp                 # Main application entry point
├── controllerMain.cpp       # Main controller logic
├── camera_biometrics.h      # Camera and biometric interface
├── content_database.*       # Content classification system
├── demoApp.cpp             # Demo application
├── testCamera01.cpp        # Camera testing utilities
├── analysisScript01.py     # Python analysis scripts
├── biometric_data.csv      # Sample biometric data
└── personal_ai_detector.pkl # Pre-trained ML model
```

## Getting Started

### Prerequisites

- C++ compiler with C++17 support
- OpenCV library
- Python 3.x (for analysis scripts)
- Eigen library

### Building

```bash
# Compile the main application
g++ -std=c++17 main.cpp -o deepfake_simulator `pkg-config --cflags --libs opencv4`

# Or compile individual components
g++ -std=c++17 controllerMain.cpp -o controller
```

### Usage

1. Run the main application:
   ```bash
   ./deepfake_simulator
   ```

2. Use the demo application:
   ```bash
   ./demoApp
   ```

3. Run analysis scripts:
   ```bash
   python3 analysisScript01.py
   ```

## Data Format

### Biometric Data
- **heart_rate**: Beats per minute
- **stress_level**: Normalized stress indicator (0-1)
- **is_human**: Boolean indicating human vs AI content
- **content_id**: Unique identifier for content samples

### Content Database
- **humanContent**: Sample human-written text
- **aiContent**: Sample AI-generated text
- **ContentSample**: Struct containing text and classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Project Focus

This project is part of the CS Girlies Hackathon 2024. See individual file headers for specific licensing information.
"How can you replace AI before it replaces you?"
New Ideas that AI can't do for you

## Acknowledgments

- OpenCV community for computer vision tools
- Eigen library for linear algebra operations
- Hackathon organizers and participants
