#pragma once

#include "camera.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"

struct Landmark {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

class Landmarker {
public:
    ~Landmarker();

    bool open(const std::string& model_path);
    bool detect(
        const CameraFrame& frame,
        std::vector<Landmark>& landmarks
    );

    const std::string& last_error() const;

private:
    std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarker> detector;
    std::int64_t last_timestamp_ms = -1;
    std::string error_message;
};
