#pragma once

#include <filesystem>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace FileLock {

struct FdGuard {
  int fd;
  ~FdGuard() {
    if (fd >= 0) {
      flock(fd, LOCK_UN);
      close(fd);
    }
  }
};

inline auto try_open_file(const std::filesystem::path path, size_t tries = 3) {
  for (auto i = 0; i < tries; ++i) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }
    if (flock(fd, LOCK_SH) == -1) {
      close(fd);
      throw std::runtime_error{"Agent: could not lock file: " + path.string()};
    }
    std::string fd_path = "/proc/self/fd/" + std::to_string(fd);
    std::fstream file{fd_path};
    if (file) {
      return std::make_pair(std::move(file), fd);
    }
    flock(fd, LOCK_UN);
    close(fd);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  throw std::runtime_error{"Agent: could not open file at: " + path.string()};
  return std::make_pair(std::move(std::fstream{}), -1);
}

} // namespace FileLock