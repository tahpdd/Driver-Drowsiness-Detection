#include "mar.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>

bool Mar::open(const std::string& model_path) {
    error_message.clear();
    session.reset();

    try {
        session = std::make_unique<Ort::Session>(environment, model_path.c_str(), session_options);

        if (session->GetInputCount() == 0 || session->GetOutputCount() == 0) {
            error_message = "TCN model does not have an input or output";
            session.reset();
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr model_input_name = session->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr model_output_name = session->GetOutputNameAllocated(0, allocator);
        input_name = model_input_name.get();
        output_name = model_output_name.get();
    } catch (const Ort::Exception& error) {
        error_message = error.what();
        session.reset();
        return false;
    }

    return true;
}

bool Mar::update(const std::vector<Landmark>& landmarks, MarResult& result) {
    result = MarResult{};

    if (!session) {
        error_message = "TCN model is not open";
        return false;
    }

    if (landmarks.size() <= 317) {
        recording = false;
        remaining_end_check_frames = mar_end_check_frames;
        recorded_values.clear();
        return true;
    }

    result.value = compute_mar(landmarks);

    if (!recording && result.value > 0.2) {
        recording = true;
        remaining_end_check_frames = mar_end_check_frames;
        recorded_values.clear();
        recorded_values.push_back(result.value);
    } else if (recording) {
        recorded_values.push_back(result.value);

        if (result.value < 0.2) {
            --remaining_end_check_frames;

            if (remaining_end_check_frames == 0) {
                int recorded_frame_count = static_cast<int>(recorded_values.size());
                result.recorded_frames = recorded_frame_count;

                if (recorded_frame_count >= minimum_mar_recorded_frames) {
                    if (!run_model(recorded_values, result)) {
                        return false;
                    }
                }

                recording = false;
                remaining_end_check_frames = mar_end_check_frames;
                recorded_values.clear();
            }
        } else {
            remaining_end_check_frames = mar_end_check_frames;
        }
    }

    result.recording = recording;

    if (recording) {
        result.recorded_frames = static_cast<int>(recorded_values.size());
    }

    return true;
}

const std::string& Mar::last_error() const {
    return error_message;
}

double Mar::compute_mar(const std::vector<Landmark>& landmarks) const {
    double center_opening = distance(landmarks[13], landmarks[14]);
    double left_opening = distance(landmarks[82], landmarks[87]);
    double right_opening = distance(landmarks[312], landmarks[317]);
    double mouth_width = distance(landmarks[78], landmarks[308]);

    if (mouth_width < 0.000001) {
        return 0.0;
    }

    return (center_opening + left_opening + right_opening) / (3.0 * mouth_width);
}

double Mar::distance(const Landmark& first, const Landmark& second) const {
    double x_distance = first.x - second.x;
    double y_distance = first.y - second.y;
    return std::hypot(x_distance, y_distance);
}

std::vector<double> Mar::prepare_samples(const std::vector<double>& values) const {
    const int sample_count = 150;
    int value_count = static_cast<int>(values.size());

    if (values.empty()) {
        return std::vector<double>(sample_count, 0.0);
    }

    if (value_count <= sample_count) {
        std::vector<double> samples = values;
        samples.resize(sample_count, values.back());
        return samples;
    }

    std::vector<double> samples(sample_count);

    for (int index = 0; index < sample_count; ++index) {
        double position = static_cast<double>(index) * static_cast<double>(value_count - 1) / static_cast<double>(sample_count - 1);
        int left_index = static_cast<int>(position);
        int right_index = std::min(left_index + 1, value_count - 1);
        double weight = position - static_cast<double>(left_index);
        samples[index] = values[left_index] * (1.0 - weight) + values[right_index] * weight;
    }

    return samples;
}

bool Mar::run_model(const std::vector<double>& values, MarResult& result) {
    std::vector<double> samples = prepare_samples(values);
    std::vector<float> model_samples(samples.begin(), samples.end());
    std::array<std::int64_t, 3> input_shape{1, 1, 150};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, model_samples.data(), model_samples.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    Ort::RunOptions run_options;

    try {
        std::vector<Ort::Value> outputs = session->Run(run_options, input_names, &input_tensor, 1, output_names, 1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            error_message = "TCN model did not return a tensor";
            return false;
        }

        int output_size = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());

        if (output_size < 3) {
            error_message = "TCN model output has fewer than 3 values";
            return false;
        }

        const float* logits = outputs[0].GetTensorData<float>();
        double largest_logit = std::max(static_cast<double>(logits[0]), std::max(static_cast<double>(logits[1]), static_cast<double>(logits[2])));
        double probability_sum = 0.0;

        for (int index = 0; index < 3; ++index) {
            result.probabilities[index] = std::exp(static_cast<double>(logits[index]) - largest_logit);
            probability_sum += result.probabilities[index];
        }

        for (double& probability : result.probabilities) {
            probability /= probability_sum;
        }

        result.prediction = 0;

        for (int index = 1; index < 3; ++index) {
            if (result.probabilities[index] > result.probabilities[result.prediction]) {
                result.prediction = index;
            }
        }

        result.label = prediction_label(result.prediction);
        result.model_ran = true;
    } catch (const Ort::Exception& error) {
        error_message = error.what();
        return false;
    } catch (const std::exception& error) {
        error_message = error.what();
        return false;
    }

    return true;
}

std::string Mar::prediction_label(int prediction) const {
    if (prediction == 0) {
        return "Normal";
    }

    if (prediction == 1) {
        return "Talking";
    }

    if (prediction == 2) {
        return "Yawning";
    }

    return "Unknown";
}
