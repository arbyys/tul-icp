#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "noncopyable.hpp"
#include "miniaudio.h"

class AudioManager : NonCopyable {
public:
    AudioManager();
    ~AudioManager();

    bool init();
    void shutdown();

    bool load3D(const std::string& key, const std::filesystem::path& file, float min_distance, float max_distance, float volume);
    bool play3DOneShot(const std::string& key, float x, float y, float z);

    bool ensure3DLoop(const std::string& loop_id, const std::string& key, float x, float y, float z);
    void stop3DLoop(const std::string& loop_id);

    void set_listener_position(float x, float y, float z, float dir_x, float dir_y, float dir_z);
    void clean_finished_sounds();

private:
    struct SoundSettings {
        float min_distance = 1.0f;
        float max_distance = 100.0f;
        float volume = 1.0f;
    };

    ma_engine engine{};

    std::unordered_map<std::string, std::unique_ptr<ma_sound>> sound_bank;
    std::unordered_map<std::string, SoundSettings> sound_settings;
    std::vector<std::unique_ptr<ma_sound>> active_one_shots;
    std::unordered_map<std::string, std::unique_ptr<ma_sound>> active_loops;

    bool init_sound_copy(ma_sound* source, std::unique_ptr<ma_sound>& destination);
    static void apply_3d_defaults(ma_sound* sound, const SoundSettings& settings, float x, float y, float z, bool looping);
};
