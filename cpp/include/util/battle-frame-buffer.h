#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct BattleFrameBuffer {

  std::vector<char> buffer;
  size_t write_index;

  BattleFrameBuffer(size_t size) : buffer{}, write_index{} {
    buffer.resize(size);
  }

  void save_to_disk(const std::filesystem::path &dir,
                    std::atomic<uint64_t> &counter) {
    if (write_index == 0) {
      return;
    }
    const auto filename =
        std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) +
        ".battle.data";
    const std::filesystem::path full_path = dir / filename;
    std::ofstream out(full_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cout << "Failed to write buffer to " << full_path << '\n';
      return;
    }
    out.write(reinterpret_cast<const char *>(buffer.data()),
              static_cast<std::streamsize>(write_index));
    clear();
  }

  void clear() {
    std::fill(buffer.begin(), buffer.end(), 0);
    write_index = 0;
  }

  void write_frames(const auto &training_frames) {
    const auto n_bytes_frames = training_frames.n_bytes();
    training_frames.write(buffer.data() + write_index);
    write_index += n_bytes_frames;
  }
};
