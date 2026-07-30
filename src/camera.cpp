#include "camera.h"

#include <cerrno>
#include <cstring>
#include <sstream>
#include <thread>
#include <utility>

Camera::Camera(std::string device, int width, int height, int fps)
    : device(device),
      width(width),
      height(height),
      fps(fps),
      frame_size(0),
      process(nullptr)
{
    if (width > 0 && height > 0) {
        frame_size = width * height * 3;
    }
}

Camera::~Camera() {
    close();
}

bool Camera::open() {
    if (is_open()) {
        return true;
    }

    if (device.empty() || width <= 0 || height <= 0 || fps <= 0) {
        error_message = "Invalid camera config.";
        return false;
    }

    process = popen(build_command().c_str(), "r");
    if (process == nullptr) {
        error_message = "Can not start FFmpeg: ";
        error_message += std::strerror(errno);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        frame_ready = false;
    }

    error_message.clear();
    running = true;
    reader_thread = std::thread(&Camera::read_loop, this);
    return true;
}

bool Camera::read_frame(CameraFrame& frame) {
    if (!is_open()) {
        error_message = "Camera is not open.";
        return false;
    }

    std::unique_lock<std::mutex> lock(frame_mutex);

    while (!frame_ready && running) {
        frame_condition.wait(lock);
    }

    if (!frame_ready) {
        return false;
    }

    std::swap(frame, latest_frame);
    frame_ready = false;
    return true;
}

bool Camera::read_process_frame(CameraFrame& frame) {
    frame.width = width;
    frame.height = height;
    frame.rgb.resize(frame_size);

    int total_bytes = 0;
    while (total_bytes < frame_size) {
        int bytes_read = static_cast<int>(std::fread(frame.rgb.data() + total_bytes, 1, frame_size - total_bytes, process));

        if (bytes_read == 0) {
            if (std::ferror(process) != 0) {
                error_message = "Could not read FFmpeg output: ";
                error_message += std::strerror(errno);
            } else {
                error_message = "FFmpeg stopped before a complete frame was read.";
            }
            return false;
        }

        total_bytes += bytes_read;
    }

    error_message.clear();
    return true;
}

void Camera::close() {
    running = false;
    frame_condition.notify_all();

    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    if (process != nullptr) {
        pclose(process);
        process = nullptr;
    }
}

bool Camera::is_open() const {
    return process != nullptr;
}

const std::string& Camera::last_error() const {
    return error_message;
}

void Camera::read_loop() {
    CameraFrame frame;

    while (running) {
        if (!read_process_frame(frame)) {
            running = false;
            frame_condition.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            std::swap(latest_frame, frame);
            frame_ready = true;
        }

        frame_condition.notify_one();
    }
}

std::string Camera::build_command() const {
    std::ostringstream command;
    command
        << "ffmpeg"
        << " -loglevel quiet"
        << " -f v4l2"
        << " -input_format mjpeg"
        << " -video_size " << width << "x" << height
        << " -framerate " << fps
        << " -i " << device
        << " -f rawvideo"
        << " -pix_fmt rgb24"
        << " -";
    return command.str();
}
