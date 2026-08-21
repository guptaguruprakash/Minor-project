#pragma once

#include <Arduino.h>
#include <math.h>

enum RepExercise : uint8_t {
  REP_EXERCISE_SQUAT = 0,
  REP_EXERCISE_BICEP = 1,
};

enum RepState : uint8_t {
  REP_IDLE = 0,
  REP_MOVING_OUT = 1,
  REP_RETURNING = 2,
};

struct RepCounterConfig {
  float smoothingAlpha = 0.20f;
  float movementThreshold = 12.0f;
  float peakThreshold = 20.0f;
  float hysteresisThreshold = 5.0f;
  uint32_t minimumRepDurationMs = 400;
  uint32_t minimumTimeBetweenRepsMs = 700;
  uint32_t baselineUpdateIntervalMs = 1000;
};

class RepCounter {
 public:
  explicit RepCounter(const RepCounterConfig& config = RepCounterConfig())
      : config_(config) {
    reset();
  }

  void reset() {
    state_ = REP_IDLE;
    exercise_ = REP_EXERCISE_SQUAT;
    repetitions_ = 0;
    filteredSignal_ = 0.0f;
    baseline_ = 0.0f;
    peakExcursion_ = 0.0f;
    initialized_ = false;
    peakSeen_ = false;
    movementStartedAt_ = 0;
    lastRepetitionAt_ = 0;
    lastBaselineUpdateAt_ = 0;
  }

  void setExercise(RepExercise exercise) {
    if (exercise_ != exercise) {
      exercise_ = exercise;
      state_ = REP_IDLE;
      peakSeen_ = false;
    }
  }

  uint16_t update(float signal, uint32_t timestampMs, bool classifierActive) {
    if (!initialized_) {
      filteredSignal_ = signal;
      baseline_ = signal;
      initialized_ = true;
      lastBaselineUpdateAt_ = timestampMs;
      return repetitions_;
    }

    filteredSignal_ += config_.smoothingAlpha * (signal - filteredSignal_);
    const float excursion = fabsf(filteredSignal_ - baseline_);
    const bool canStart = classifierActive &&
                          excursion >= config_.movementThreshold;
    const bool hasReturned = excursion <=
                             config_.movementThreshold - config_.hysteresisThreshold;

    if (state_ == REP_IDLE) {
      if (timestampMs - lastBaselineUpdateAt_ >= config_.baselineUpdateIntervalMs &&
          !classifierActive) {
        baseline_ += 0.05f * (filteredSignal_ - baseline_);
        lastBaselineUpdateAt_ = timestampMs;
      }
      if (canStart && timestampMs - lastRepetitionAt_ >= config_.minimumTimeBetweenRepsMs) {
        state_ = REP_MOVING_OUT;
        movementStartedAt_ = timestampMs;
        peakExcursion_ = excursion;
        peakSeen_ = excursion >= config_.peakThreshold;
      }
      return repetitions_;
    }

    if (state_ == REP_MOVING_OUT) {
      if (excursion > peakExcursion_) {
        peakExcursion_ = excursion;
      }
      if (peakExcursion_ >= config_.peakThreshold) {
        peakSeen_ = true;
        state_ = REP_RETURNING;
      }
      return repetitions_;
    }

    if (state_ == REP_RETURNING && hasReturned) {
      const uint32_t duration = timestampMs - movementStartedAt_;
      if (peakSeen_ && duration >= config_.minimumRepDurationMs &&
          timestampMs - lastRepetitionAt_ >= config_.minimumTimeBetweenRepsMs) {
        ++repetitions_;
        lastRepetitionAt_ = timestampMs;
      }
      state_ = REP_IDLE;
      peakExcursion_ = 0.0f;
      peakSeen_ = false;
    }
    return repetitions_;
  }

  uint16_t repetitions() const { return repetitions_; }
  RepState state() const { return state_; }
  RepExercise exercise() const { return exercise_; }
  float filteredSignal() const { return filteredSignal_; }
  float baseline() const { return baseline_; }

 private:
  RepCounterConfig config_;
  RepState state_;
  RepExercise exercise_;
  uint16_t repetitions_;
  float filteredSignal_;
  float baseline_;
  float peakExcursion_;
  bool initialized_;
  bool peakSeen_;
  uint32_t movementStartedAt_;
  uint32_t lastRepetitionAt_;
  uint32_t lastBaselineUpdateAt_;
};
