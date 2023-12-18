#pragma once

#pragma warning(push, 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/script.h>
#include <torch/torch.h>
#pragma GCC diagnostic pop
#pragma warning(pop)

namespace torch {

/// This is a wrapper around torch::pickle_load() that loads from a
/// file instead of a buffer.
/// torch::pickle_load() is used to load tensors exported by
/// torch.save() in Python
torch::IValue pickle_load(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not read file " + filename);
  }
  std::vector<char> buffer(std::istreambuf_iterator<char>(in), {});
  return pickle_load(buffer);
}

/// This is a wrapper around torch::pickle_save() that saves to a
/// file instead of a buffer.
/// torch::pickle_save() is used to save tensors that can be loaded
/// by torch.load() in Python
void pickle_save(torch::IValue value, const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  auto data = pickle_save(value);
  if (!out || !out.write(data.data(), data.size())) {
    throw std::runtime_error("Failed to write file " + filename);
  }
}

}  // namespace torch
