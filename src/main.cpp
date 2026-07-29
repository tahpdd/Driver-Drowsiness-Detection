#include "camera.h"
#include "ear.h"
#include "landmarker.h"
#include "mar.h"

#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

volatile std::sig_atomic_t stop_requested = 0;

void handle_signal(int) {
    stop_requested = 1;
}

int main() {
    std::signal(SIGINT, handle_signal);

    Camera camera("/dev/video0", 1280, 720, 30);
    Landmarker landmarker;
    Ear ear;
    Mar mar;

    if (!camera.open()) {
        std::cerr << "Camera error: "
                  << camera.last_error() << '\n';
        return 1;
    }

    if (!landmarker.open("models/face_landmarker.task")) {
        std::cerr << "Landmarker error: "
                  << landmarker.last_error() << '\n';
        return 1;
    }

    if (!mar.open("models/tcn_model.onnx")) {
        std::cerr << "MAR error: "
                  << mar.last_error() << '\n';
        return 1;
    }

    CameraFrame frame;
    std::vector<Landmark> landmarks;
    EarResult ear_result;
    MarResult mar_result;
    int frame_index = 0;
    std::string tcn_label = "Waiting";

    std::cout << "\033[2J\033[H";

    while (!stop_requested) {
        if (!camera.read_frame(frame)) {
            std::cerr << "Camera error: "
                      << camera.last_error() << '\n';
            return 1;
        }

        if (stop_requested) {
            break;
        }

        ++frame_index;

        if (!landmarker.detect(frame, landmarks)) {
            std::cerr << "Landmarker error: "
                      << landmarker.last_error() << '\n';
            return 1;
        }

        ear.update(landmarks, ear_result);

        if (!mar.update(landmarks, mar_result)) {
            std::cerr << "MAR error: "
                      << mar.last_error() << '\n';
            return 1;
        }

        if (mar_result.model_ran) {
            tcn_label = mar_result.label;
        }

        std::cout << "\033[H"
                  << "Frame: " << frame_index
                  << " | Face: " << (landmarks.empty() ? "No" : "Yes")
                  << " | Landmarks: " << landmarks.size()
                  << "\033[K\n"
                  << "MAR: " << std::fixed << std::setprecision(3)
                  << mar_result.value
                  << " | Recording: " << (mar_result.recording ? "Yes" : "No")
                  << " | Recorded frames: " << mar_result.recorded_frames
                  << " | TCN: " << tcn_label
                  << "\033[K\n"
                  << "EAR: " << std::setprecision(3)
                  << ear_result.left_ear << "/"
                  << ear_result.right_ear
                  << " | Open: " << std::setprecision(1)
                  << ear_result.left_open_percent << "%/"
                  << ear_result.right_open_percent << "%"
                  << " | Eyes closed: " << (ear_result.eyes_closed ? "Yes" : "No")
                  << " | Closed frames: " << ear_result.closed_frame_count
                  << "\033[K\n"
                  << std::flush;
    }

    std::cout << "Stopping...\n";
    return 0;
}
