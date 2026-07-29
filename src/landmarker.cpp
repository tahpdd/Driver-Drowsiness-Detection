#include "landmarker.h"

#include <chrono>
#include <utility>

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

Landmarker::~Landmarker() {
    if (detector != nullptr) {
        detector->Close().IgnoreError();
    }
}

bool Landmarker::open(const std::string& model_path) {
    auto options = std::make_unique<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();

    options->base_options.model_asset_path = model_path;
    options->running_mode =
        mediapipe::tasks::vision::core::RunningMode::VIDEO;
    options->num_faces = 1;

    auto result = mediapipe::tasks::vision::face_landmarker::FaceLandmarker::Create(std::move(options));

    if (!result.ok()) {
        error_message = result.status().ToString();
        return false;
    }

    detector = std::move(result.value());
    error_message.clear();
    return true;
}

bool Landmarker::detect(const CameraFrame& frame, std::vector<Landmark>& landmarks) {
    landmarks.clear();

    if (detector == nullptr) {
        error_message = "Face landmarker is not open.";
        return false;
    }

    auto image_result = mediapipe::tasks::vision::CreateImageFromBuffer(mediapipe::ImageFormat::SRGB, frame.rgb.data(), frame.width, frame.height);

    if (!image_result.ok()) {
        error_message = image_result.status().ToString();
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    std::int64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    if (timestamp_ms <= last_timestamp_ms) {
        timestamp_ms = last_timestamp_ms + 1;
    }
    last_timestamp_ms = timestamp_ms;

    auto result = detector->DetectForVideo(std::move(image_result.value()), timestamp_ms);

    if (!result.ok()) {
        error_message = result.status().ToString();
        return false;
    }

    if (!result->face_landmarks.empty()) {
        const auto& face = result->face_landmarks.front().landmarks;
        landmarks.reserve(face.size());

        for (const auto& point : face) {
            landmarks.push_back({point.x, point.y, point.z});
        }
    }

    error_message.clear();
    return true;
}

const std::string& Landmarker::last_error() const {
    return error_message;
}
