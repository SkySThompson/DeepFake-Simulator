// content_database.h
#ifndef CONTENT_DATABASE_H
#define CONTENT_DATABASE_H

#include <vector>
#include <string>
using namespace std;

class ContentDatabase {
public:
    vector<string> humanContent = {
        "The old man sat by the window, watching the rain trace lazy paths down the glass. Each drop seemed to carry a memory...",
        "Local bakery owner Maria Rodriguez has been serving the community for thirty years. Her secret? 'I put love in every loaf,' she says with a smile...",
        "Scientists at MIT have discovered a new method for water purification using coffee grounds. The breakthrough came after years of experimentation...",
        // Add 20+ human-written samples
    };
    
    vector<string> aiContent = {
        "As an AI language model, I can help you understand the complex relationship between technology and society in our modern world...",
        "The implementation of advanced algorithms has revolutionized the way we process information in today's digital landscape...",
        "Certainly! Here's a comprehensive overview of the topic you've requested, breaking down the key components and their interconnected relationships...",
        // Add 20+ AI-generated samples  
    };
    
    struct ContentSample {
        string text;
        bool isHuman;  // true = human, false = AI
        int sampleId;
    };
    
    vector<ContentSample> getShuffledSamples();
};

#endif