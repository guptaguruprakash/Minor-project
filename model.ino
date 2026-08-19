#include "knn_model.h"

void setupDataCollection();
void loopDataCollection();

void setup() {
	setupDataCollection();
}

void loop() {
	loopDataCollection();
}

uint8_t predictSensorFeatures(const float features[SmartGymKnn::FEATURE_COUNT]) {
	return SmartGymKnn::predict(features);
}

const char* knnExerciseName(uint8_t label) {
	switch (label) {
		case SmartGymKnn::LABEL_BICEP_BAD:
		case SmartGymKnn::LABEL_BICEP_GOOD:
			return "bicep";
		case SmartGymKnn::LABEL_BICEP_IDLE_GOOD:
			return "bicep_idle";
		case SmartGymKnn::LABEL_SQUAT_GOOD:
			return "squat";
		default:
			return "unknown";
	}
}

const char* knnPostureName(uint8_t label) {
	switch (label) {
		case SmartGymKnn::LABEL_BICEP_BAD:
			return "bad";
		case SmartGymKnn::LABEL_BICEP_GOOD:
		case SmartGymKnn::LABEL_BICEP_IDLE_GOOD:
		case SmartGymKnn::LABEL_SQUAT_GOOD:
			return label == SmartGymKnn::LABEL_BICEP_IDLE_GOOD ? "idle" : "good";
		default:
			return "unknown";
	}
}
