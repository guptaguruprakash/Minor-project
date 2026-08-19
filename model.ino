#include "knn_model.h"

uint8_t predictSensorFeatures(const float* features) {
	return SmartGymKnn::predict(features);
}

const char* knnExerciseName(uint8_t label) {
	switch (label) {
		case SmartGymKnn::LABEL_BICEP_BAD:
		case SmartGymKnn::LABEL_BICEP_GOOD:
			return "bicep";
		case SmartGymKnn::LABEL_BICEP_IDLE_GOOD:
			return "bicep_idle";
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
			return label == SmartGymKnn::LABEL_BICEP_IDLE_GOOD ? "idle" : "good";
		default:
			return "unknown";
	}
}
