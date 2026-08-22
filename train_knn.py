"""Train a compact, dependency-free KNN model for the ESP32 sketch."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


FEATURE_NAMES = (
    "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
)
CSV_HEADERS = (
    "received_at", "timestamp", "user", "exercise", "posture", "label",
    *FEATURE_NAMES,
)
DEFAULT_FILES = ("data/train.csv",)
DEFAULT_PROTOTYPES_PER_LABEL = 64
K_NEIGHBORS = 5


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if [row.get(name, "").strip() for name in CSV_HEADERS] == list(CSV_HEADERS):
                    continue
                label = row.get("label", "").strip()
                if label:
                    rows.append((label, [float(row[name]) for name in FEATURE_NAMES]))
    if not rows:
        raise ValueError("No labeled rows were found in the input CSV files.")
    return rows


def choose_prototypes(rows, per_label):
    grouped = {}
    for label, features in rows:
        grouped.setdefault(label, []).append(features)

    selected = []
    for label in sorted(grouped):
        label_rows = grouped[label]
        count = min(per_label, len(label_rows))
        if count == 1:
            indexes = [0]
        else:
            indexes = [round(index * (len(label_rows) - 1) / (count - 1)) for index in range(count)]
        selected.extend((label, label_rows[index]) for index in indexes)
    return selected


def calculate_scaling(rows):
    means = [sum(features[index] for _, features in rows) / len(rows) for index in range(len(FEATURE_NAMES))]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((features[index] - mean) ** 2 for _, features in rows) / len(rows)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def normalize(features, means, scales):
    return [(value - mean) / scale for value, mean, scale in zip(features, means, scales)]


def predict(features, training, means, scales):
    normalized = normalize(features, means, scales)
    nearest = sorted(
        (sum((value - reference) ** 2 for value, reference in zip(normalized, normalize(row, means, scales))), label)
        for label, row in training
    )[:K_NEIGHBORS]
    return Counter(label for _, label in nearest).most_common(1)[0][0]


def validate(rows, training, means, scales):
    correct = 0
    for label, features in rows:
        correct += predict(features, training, means, scales) == label
    return correct / len(rows)


def c_float(value):
    return f"{value:.8f}f"


def write_header(path, training, means, scales):
    labels = sorted({label for label, _ in training})
    label_ids = {label: index for index, label in enumerate(labels)}
    lines = [
        "#pragma once",
        "#include <Arduino.h>",
        "#include <pgmspace.h>",
        "",
        "namespace SmartGymKnn {",
        f"constexpr uint8_t FEATURE_COUNT = {len(FEATURE_NAMES)};",
        f"constexpr uint8_t K_NEIGHBORS = {K_NEIGHBORS};",
        f"constexpr uint16_t TRAINING_SAMPLE_COUNT = {len(training)};",
        "",
        "enum Label : uint8_t {",
    ]
    for label in labels:
        lines.append(f"  LABEL_{label.upper()} = {label_ids[label]},")
    lines.extend(["};", "", "constexpr const char* LABEL_NAMES[] = {"])
    lines.extend(f'  "{label}",' for label in labels)
    lines.extend([
        "};",
        f"constexpr uint8_t LABEL_COUNT = {len(labels)};",
        "",
        "const float FEATURE_MEANS[FEATURE_COUNT] PROGMEM = {",
    ])
    lines.extend(f"  {c_float(value)}," for value in means)
    lines.extend(["};", "", "const float FEATURE_SCALES[FEATURE_COUNT] PROGMEM = {"])
    lines.extend(f"  {c_float(value)}," for value in scales)
    lines.extend(["};", "", "const float TRAINING_FEATURES[TRAINING_SAMPLE_COUNT][FEATURE_COUNT] PROGMEM = {"])
    for label, features in training:
        normalized = normalize(features, means, scales)
        lines.append("  {" + ", ".join(c_float(value) for value in normalized) + "},")
    lines.extend(["};", "", "const uint8_t TRAINING_LABELS[TRAINING_SAMPLE_COUNT] PROGMEM = {"])
    lines.extend("  " + ", ".join(str(label_ids[label]) for label, _ in training[i:i + 16]) + "," for i in range(0, len(training), 16))
    lines.extend([
        "};",
        "",
        "inline uint8_t predict(const float features[FEATURE_COUNT]) {",
        "  float nearestDistances[K_NEIGHBORS];",
        "  uint8_t nearestLabels[K_NEIGHBORS];",
        "  for (uint8_t neighbor = 0; neighbor < K_NEIGHBORS; ++neighbor) {",
        "    nearestDistances[neighbor] = INFINITY;",
        "    nearestLabels[neighbor] = 0;",
        "  }",
        "",
        "  for (uint16_t sample = 0; sample < TRAINING_SAMPLE_COUNT; ++sample) {",
        "    float distance = 0.0f;",
        "    for (uint8_t feature = 0; feature < FEATURE_COUNT; ++feature) {",
        "      const float value = (features[feature] - pgm_read_float(&FEATURE_MEANS[feature])) / pgm_read_float(&FEATURE_SCALES[feature]);",
        "      const float difference = value - pgm_read_float(&TRAINING_FEATURES[sample][feature]);",
        "      distance += difference * difference;",
        "    }",
        "    uint8_t insertion = K_NEIGHBORS;",
        "    for (uint8_t neighbor = 0; neighbor < K_NEIGHBORS; ++neighbor) {",
        "      if (distance < nearestDistances[neighbor]) {",
        "        insertion = neighbor;",
        "        break;",
        "      }",
        "    }",
        "    if (insertion < K_NEIGHBORS) {",
        "      for (uint8_t neighbor = K_NEIGHBORS - 1; neighbor > insertion; --neighbor) {",
        "        nearestDistances[neighbor] = nearestDistances[neighbor - 1];",
        "        nearestLabels[neighbor] = nearestLabels[neighbor - 1];",
        "      }",
        "      nearestDistances[insertion] = distance;",
        "      nearestLabels[insertion] = pgm_read_byte(&TRAINING_LABELS[sample]);",
        "    }",
        "  }",
        "",
        "  uint8_t votes[sizeof(LABEL_NAMES) / sizeof(LABEL_NAMES[0])] = {};",
        "  for (uint8_t neighbor = 0; neighbor < K_NEIGHBORS; ++neighbor) {",
        "    ++votes[nearestLabels[neighbor]];",
        "  }",
        "  uint8_t bestLabel = 0;",
        "  for (uint8_t label = 1; label < sizeof(LABEL_NAMES) / sizeof(LABEL_NAMES[0]); ++label) {",
        "    if (votes[label] > votes[bestLabel]) {",
        "      bestLabel = label;",
        "    }",
        "  }",
        "  return bestLabel;",
        "}",
        "",
        "inline const char* labelName(uint8_t label) {",
        "  return label < sizeof(LABEL_NAMES) / sizeof(LABEL_NAMES[0]) ? LABEL_NAMES[label] : \"unknown\";",
        "}",
        "}  // namespace SmartGymKnn",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("knn_model.h"))
    parser.add_argument("--prototypes-per-label", type=int, default=DEFAULT_PROTOTYPES_PER_LABEL)
    parser.add_argument("files", nargs="*", type=Path, default=[Path(path) for path in DEFAULT_FILES])
    args, _ = parser.parse_known_args()

    rows = load_rows(args.files)
    training = choose_prototypes(rows, args.prototypes_per_label)
    means, scales = calculate_scaling(training)
    write_header(args.output, training, means, scales)

    print(f"Loaded {len(rows)} rows from {len(args.files)} files")
    print("Labels:", dict(sorted(Counter(label for label, _ in rows).items())))
    print(f"Embedded prototypes: {len(training)} ({args.prototypes_per_label} per label maximum)")
    print(f"Training-set accuracy: {validate(training, training, means, scales):.2%}")


if __name__ == "__main__":
    main()