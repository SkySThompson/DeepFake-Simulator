#include "content_database.h"
#include <algorithm>
#include <random>
#include <chrono>

vector<ContentDatabase::ContentSample> ContentDatabase::getShuffledSamples() {
    vector<ContentSample> samples;
    int id = 0;

    for (const auto& text : humanContent) {
        samples.push_back({text, true, id++});
    }
    for (const auto& text : aiContent) {
        samples.push_back({text, false, id++});
    }

    // Shuffle samples with a random seed based on time
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(samples.begin(), samples.end(), std::default_random_engine(seed));

    return samples;
}
