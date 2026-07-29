#pragma once

#include "landmarker.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

constexpr int minimum_mar_recorded_frames = 90;
constexpr int mar_end_check_frames = 15;

struct MarResult {
    double value = 0.0;
    bool recording = false;
    int recorded_frames = 0;
    bool model_ran = false;
    int prediction = -1;
    std::string label;
    std::array<double, 3> probabilities{0.0, 0.0, 0.0};
};

class Mar {
public:
    bool open(const std::string& model_path);
    bool update(
        const std::vector<Landmark>& landmarks,
        MarResult& result
    );

    const std::string& last_error() const;

private:
    double compute_mar(const std::vector<Landmark>& landmarks) const;
    double distance(const Landmark& first, const Landmark& second) const;
    std::vector<double> prepare_samples(const std::vector<double>& values) const;
    bool run_model(
        const std::vector<double>& values,
        MarResult& result
    );
    std::string prediction_label(int prediction) const;

    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "ddd"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::string error_message;
    bool recording = false;
    int remaining_end_check_frames = mar_end_check_frames;
    std::vector<double> recorded_values;
};
