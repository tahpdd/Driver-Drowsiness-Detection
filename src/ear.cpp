#include "ear.h"

#include <algorithm>
#include <cmath>

void Ear::update(const std::vector<Landmark>& landmarks, EarResult& result) {
    result = EarResult{};

    if (landmarks.size() <= 387) {
        closed_frame_count = 0;
        return;
    }

    result.left_ear = compute_eye_ear(landmarks, 385, 373, 387, 380, 362, 263);
    result.right_ear = compute_eye_ear(landmarks, 160, 144, 158, 153, 133, 33);

    add_to_window(left_values, result.left_ear);
    add_to_window(right_values, result.right_ear);

    double left_open_reference = 0.0;
    double left_close_reference = 0.0;
    double right_open_reference = 0.0;
    double right_close_reference = 0.0;

    calculate_references(left_values, left_open_reference, left_close_reference);
    calculate_references(right_values, right_open_reference, right_close_reference);

    result.left_open_percent = calculate_open_percent(result.left_ear, left_open_reference, left_close_reference);
    result.right_open_percent = calculate_open_percent(result.right_ear, right_open_reference, right_close_reference);
    result.eyes_closed = result.left_open_percent < ear_closed_threshold_percent && result.right_open_percent < ear_closed_threshold_percent;

    if (result.eyes_closed) {
        ++closed_frame_count;
    } else {
        closed_frame_count = 0;
    }

    result.closed_frame_count = closed_frame_count;
}

double Ear::compute_eye_ear(const std::vector<Landmark>& landmarks, int top_left, int bottom_left, int top_right, int bottom_right, int corner_left, int corner_right) const {
    double first_vertical = std::abs(landmarks[top_left].y - landmarks[bottom_left].y);
    double second_vertical = std::abs(landmarks[top_right].y - landmarks[bottom_right].y);

    double x_distance = landmarks[corner_left].x - landmarks[corner_right].x;
    double y_distance = landmarks[corner_left].y - landmarks[corner_right].y;
    double z_distance = landmarks[corner_left].z - landmarks[corner_right].z;

    double horizontal = std::sqrt(x_distance * x_distance + y_distance * y_distance + z_distance * z_distance);

    if (horizontal < 0.000001) {
        return 0.0;
    }

    return (first_vertical + second_vertical) / (2.0 * horizontal);
}

void Ear::calculate_references(const std::vector<double>& values, double& open_reference, double& close_reference) const {
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    int value_count = static_cast<int>(sorted_values.size());
    int open_index = static_cast<int>((value_count - 1) * ear_open_reference_percentile / 100.0);
    int close_index = static_cast<int>((value_count - 1) * ear_close_reference_percentile / 100.0);
    
    open_reference = sorted_values[open_index];
    close_reference = sorted_values[close_index];
}

double Ear::calculate_open_percent(double ear, double open_reference, double close_reference) const {
    double reference_distance = open_reference - close_reference;

    if (reference_distance < 0.000001) {
        return 100.0;
    }

    double open_percent = 100.0 * (ear - close_reference) / reference_distance;
    return std::clamp(open_percent, 0.0, 100.0);
}

void Ear::add_to_window(std::vector<double>& values, double ear) {
    values.push_back(ear);

    if (static_cast<int>(values.size()) > ear_reference_window_frames) {
        values.erase(values.begin());
    }
}
