#include "audio_manager.hpp"

#include <algorithm>

AudioManager::AudioManager() = default;

AudioManager::~AudioManager() {
}

bool AudioManager::init() {
    return ma_engine_init(nullptr, &engine) == MA_SUCCESS;
}

void AudioManager::shutdown() {
    for (auto& [_, loop_sound] : active_loops) {
        if (loop_sound) {
            ma_sound_stop(loop_sound.get());
            ma_sound_uninit(loop_sound.get());
        }
    }
    active_loops.clear();

    for (auto& one_shot : active_one_shots) {
        if (one_shot) {
            ma_sound_stop(one_shot.get());
            ma_sound_uninit(one_shot.get());
        }
    }
    active_one_shots.clear();

    for (auto& [_, sound] : sound_bank) {
        if (sound) {
            ma_sound_uninit(sound.get());
        }
    }
    sound_bank.clear();
    sound_settings.clear();

    ma_engine_uninit(&engine);
}

bool AudioManager::load3D(const std::string& key, const std::filesystem::path& file, float min_distance, float max_distance, float volume) {
    auto loaded_sound = std::make_unique<ma_sound>();
    if (ma_sound_init_from_file(&engine, file.string().c_str(), MA_SOUND_FLAG_ASYNC, nullptr, nullptr, loaded_sound.get()) != MA_SUCCESS) {
        return false;
    }

    sound_settings[key] = SoundSettings{ min_distance, max_distance, volume };
    sound_bank[key] = std::move(loaded_sound);
    return true;
}

bool AudioManager::play3DOneShot(const std::string& key, float x, float y, float z) {
    if (!sound_bank.contains(key)) {
        return false;
    }

    auto one_shot = std::make_unique<ma_sound>();
    if (!init_sound_copy(sound_bank.at(key).get(), one_shot)) {
        return false;
    }

    apply_3d_defaults(one_shot.get(), sound_settings.at(key), x, y, z, false);
    if (ma_sound_start(one_shot.get()) != MA_SUCCESS) {
        ma_sound_uninit(one_shot.get());
        return false;
    }

    active_one_shots.push_back(std::move(one_shot));
    return true;
}

bool AudioManager::ensure3DLoop(const std::string& loop_id, const std::string& key, float x, float y, float z) {
    if (!sound_bank.contains(key)) {
        return false;
    }

    if (active_loops.contains(loop_id)) {
        ma_sound_set_position(active_loops.at(loop_id).get(), x, y, z);
        if (!ma_sound_is_playing(active_loops.at(loop_id).get())) {
            ma_sound_start(active_loops.at(loop_id).get());
        }
        return true;
    }

    auto loop_sound = std::make_unique<ma_sound>();
    if (!init_sound_copy(sound_bank.at(key).get(), loop_sound)) {
        return false;
    }

    apply_3d_defaults(loop_sound.get(), sound_settings.at(key), x, y, z, true);
    if (ma_sound_start(loop_sound.get()) != MA_SUCCESS) {
        ma_sound_uninit(loop_sound.get());
        return false;
    }

    active_loops[loop_id] = std::move(loop_sound);
    return true;
}

void AudioManager::stop3DLoop(const std::string& loop_id) {
    if (!active_loops.contains(loop_id)) {
        return;
    }

    auto& loop_sound = active_loops.at(loop_id);
    ma_sound_stop(loop_sound.get());
    ma_sound_uninit(loop_sound.get());
    active_loops.erase(loop_id);
}

void AudioManager::set_listener_position(float x, float y, float z, float dir_x, float dir_y, float dir_z) {
    ma_engine_listener_set_position(&engine, 0, x, y, z);
    ma_engine_listener_set_direction(&engine, 0, dir_x, dir_y, dir_z);
}

void AudioManager::clean_finished_sounds() {
    active_one_shots.erase(
        std::remove_if(active_one_shots.begin(), active_one_shots.end(), [](const std::unique_ptr<ma_sound>& sound) {
            if (!sound) {
                return true;
            }
            if (!ma_sound_is_playing(sound.get()) || ma_sound_at_end(sound.get())) {
                ma_sound_uninit(sound.get());
                return true;
            }
            return false;
        }),
        active_one_shots.end()
    );
}

bool AudioManager::init_sound_copy(ma_sound* source, std::unique_ptr<ma_sound>& destination) {
    if (ma_sound_init_copy(&engine, source, MA_SOUND_FLAG_ASYNC, nullptr, destination.get()) != MA_SUCCESS) {
        return false;
    }

    ma_sound_seek_to_pcm_frame(destination.get(), 0);
    return true;
}

void AudioManager::apply_3d_defaults(ma_sound* sound, const SoundSettings& settings, float x, float y, float z, bool looping) {
    ma_sound_set_spatialization_enabled(sound, MA_TRUE);
    ma_sound_set_looping(sound, looping ? MA_TRUE : MA_FALSE);
    ma_sound_set_min_distance(sound, settings.min_distance);
    ma_sound_set_max_distance(sound, settings.max_distance);
    ma_sound_set_volume(sound, settings.volume);
    ma_sound_set_position(sound, x, y, z);
}
