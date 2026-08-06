#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace adi {

enum class GgufValueType : std::uint32_t {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

enum class GgmlType : std::uint32_t {
    f32 = 0,
    f16 = 1,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 30,
};

struct GgufMetadata {
    std::string_view key;
    GgufValueType type;
    std::uint64_t value_offset;
};

struct GgufTensor {
    std::string_view name;
    std::vector<std::uint64_t> dimensions;
    GgmlType type;
    std::uint64_t offset;
    std::uint64_t elements;
    std::uint64_t bytes;
};

class GgufFile {
  public:
    explicit GgufFile(const std::filesystem::path &path);
    ~GgufFile();

    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;
    GgufFile(GgufFile &&other) noexcept;
    GgufFile &operator=(GgufFile &&other) noexcept;

    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
    [[nodiscard]] std::uint64_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] std::span<const GgufMetadata> metadata() const noexcept { return metadata_; }
    [[nodiscard]] std::span<const GgufTensor> tensors() const noexcept { return tensors_; }

    [[nodiscard]] const GgufMetadata *find_metadata(std::string_view key) const noexcept;
    [[nodiscard]] const GgufTensor *find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] std::optional<std::uint64_t> integer(std::string_view key) const;
    [[nodiscard]] std::optional<double> number(std::string_view key) const;
    [[nodiscard]] std::optional<bool> boolean(std::string_view key) const;
    [[nodiscard]] std::optional<std::string_view> string(std::string_view key) const;
    [[nodiscard]] std::vector<std::string_view> string_array(std::string_view key) const;
    [[nodiscard]] std::vector<std::uint64_t> integer_array(std::string_view key) const;
    [[nodiscard]] std::span<const std::byte> tensor_data(const GgufTensor &tensor) const;

  private:
    void close() noexcept;
    [[nodiscard]] const std::byte *at(std::uint64_t offset, std::uint64_t size) const;

#ifdef _WIN32
    void *file_handle_ = nullptr;
    void *mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
    const std::byte *mapping_ = nullptr;
    std::uint64_t file_size_ = 0;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::vector<GgufMetadata> metadata_;
    std::vector<GgufTensor> tensors_;
};

[[nodiscard]] std::string_view ggml_type_name(GgmlType type) noexcept;

} // namespace adi
