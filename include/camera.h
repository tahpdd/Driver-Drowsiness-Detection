#pragma once

#include <cstdint>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct CameraFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

class Camera {
public:
    Camera(
        std::string camera_device,
        int frame_width,
        int frame_height,
        int frame_rate
    );
    ~Camera();

    bool open();
    bool read_frame(CameraFrame& frame);
    void close();

    bool is_open() const;
    const std::string& last_error() const;

private:
    std::string build_command() const;
    bool read_process_frame(CameraFrame& frame);
    void read_loop();

    std::string device;
    int width;
    int height;
    int fps;
    int frame_size;
    FILE* process;
    std::string error_message;

    std::atomic<bool> running{false};
    std::thread reader_thread;
    std::mutex frame_mutex;
    std::condition_variable frame_condition;
    CameraFrame latest_frame;
    bool frame_ready = false;
};
