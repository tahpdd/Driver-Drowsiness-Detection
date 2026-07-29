#pragma once

#include "landmarker.h"

#include <vector>

constexpr int ear_reference_window_frames = 900;
constexpr double ear_open_reference_percentile = 90.0;
constexpr double ear_close_reference_percentile = 1.0;
constexpr double ear_closed_threshold_percent = 25.0;

struct EarResult {
    double left_ear = 0.0;
    double right_ear = 0.0;
    double left_open_percent = 0.0;
    double right_open_percent = 0.0;
    bool eyes_closed = false;
    int closed_frame_count = 0;
};

class Ear {
public:
    void update(
        const std::vector<Landmark>& landmarks,
        EarResult& result
    );

private:
    double compute_eye_ear(
        const std::vector<Landmark>& landmarks,
        int top_left,
        int bottom_left,
        int top_right,
        int bottom_right,
        int corner_left,
        int corner_right
    ) const;

    void calculate_references(
        const std::vector<double>& values,
        double& open_reference,
        double& close_reference
    ) const;

    double calculate_open_percent(double ear, double open_reference, double close_reference) const;
    void add_to_window(std::vector<double>& values, double ear);

    std::vector<double> left_values;
    std::vector<double> right_values;
    
    int closed_frame_count = 0;
};
