#include "knn_model.h"

uint8_t predictSensorFeatures(const float* features) {
	return SmartGymKnn::predict(features);
}

const char* knnExerciseName(uint8_t label) {
	switch (label) {
		case SmartGymKnn::LABEL_BICEP_BAD:
		case SmartGymKnn::LABEL_BICEP_GOOD:
			return "bicep";
		case SmartGymKnn::LABEL_BICEP_IDLE:
			return "bicep";
		case SmartGymKnn::LABEL_PUSHUP_BAD:
		case SmartGymKnn::LABEL_PUSHUP_GOOD:
		case SmartGymKnn::LABEL_PUSHUP_IDLE:
			return "pushup";
		case SmartGymKnn::LABEL_SQUAT_BAD:
		case SmartGymKnn::LABEL_SQUAT_GOOD:
		case SmartGymKnn::LABEL_SQUAT_IDLE:
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
		case SmartGymKnn::LABEL_BICEP_IDLE:
			return label == SmartGymKnn::LABEL_BICEP_IDLE ? "idle" : "good";
		case SmartGymKnn::LABEL_PUSHUP_BAD:
		case SmartGymKnn::LABEL_SQUAT_BAD:
			return "bad";
		case SmartGymKnn::LABEL_PUSHUP_GOOD:
		case SmartGymKnn::LABEL_PUSHUP_IDLE:
			return label == SmartGymKnn::LABEL_PUSHUP_IDLE ? "idle" : "good";
		case SmartGymKnn::LABEL_SQUAT_GOOD:
			return "good";
		case SmartGymKnn::LABEL_SQUAT_IDLE:
			return "idle";
		default:
			return "unknown";
	}
}
