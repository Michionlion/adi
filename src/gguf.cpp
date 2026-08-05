#include "adi/gguf.hpp"

#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace adi {
namespace {

constexpr std::uint32_t gguf_magic = 0x46554747U;
constexpr std::uint32_t supported_version = 3;
constexpr std::uint32_t max_dimensions = 4;

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error("GGUF: " + std::string(message));
}

[[noreturn]] void fail_errno(std::string_view action, const std::filesystem::path &path) {
    throw std::runtime_error(
        std::string(action) + " '" + path.string() + "': " + std::strerror(errno));
}

class Cursor {
  public:
    Cursor(const std::byte *data, std::uint64_t size) : data_(data), size_(size) {}

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T));
        T value;
        std::memcpy(&value, data_ + position_, sizeof(T));
        position_ += sizeof(T);
        if constexpr (std::endian::native == std::endian::big) {
            fail("big-endian hosts are not supported");
        }
        return value;
    }

    std::string_view read_string() {
        const auto length = read<std::uint64_t>();
        require(length);
        const auto *characters = reinterpret_cast<const char *>(data_ + position_);
        position_ += length;
        return {characters, static_cast<std::size_t>(length)};
    }

    void skip(std::uint64_t count) {
        require(count);
        position_ += count;
    }

    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }
    [[nodiscard]] std::uint64_t remaining() const noexcept { return size_ - position_; }

  private:
    void require(std::uint64_t count) const {
        if (count > size_ || position_ > size_ - count) {
            fail("truncated file at byte " + std::to_string(position_));
        }
    }

    const std::byte *data_;
    std::uint64_t size_;
    std::uint64_t position_ = 0;
};

std::uint64_t scalar_size(GgufValueType type) {
    switch (type) {
    case GgufValueType::u8:
    case GgufValueType::i8:
    case GgufValueType::boolean:
        return 1;
    case GgufValueType::u16:
    case GgufValueType::i16:
        return 2;
    case GgufValueType::u32:
    case GgufValueType::i32:
    case GgufValueType::f32:
        return 4;
    case GgufValueType::u64:
    case GgufValueType::i64:
    case GgufValueType::f64:
        return 8;
    default:
        return 0;
    }
}

GgufValueType read_value_type(Cursor &cursor) {
    const auto raw = cursor.read<std::uint32_t>();
    if (raw > static_cast<std::uint32_t>(GgufValueType::f64)) {
        fail("unknown metadata value type " + std::to_string(raw));
    }
    return static_cast<GgufValueType>(raw);
}

void skip_value(Cursor &cursor, GgufValueType type, unsigned depth = 0) {
    if (depth > 8) {
        fail("metadata arrays are nested too deeply");
    }
    if (const auto size = scalar_size(type); size != 0) {
        cursor.skip(size);
        return;
    }
    if (type == GgufValueType::string) {
        (void)cursor.read_string();
        return;
    }
    if (type != GgufValueType::array) {
        fail("invalid metadata value type");
    }

    const auto element_type = read_value_type(cursor);
    const auto count = cursor.read<std::uint64_t>();
    if (const auto size = scalar_size(element_type); size != 0) {
        if (count > std::numeric_limits<std::uint64_t>::max() / size) {
            fail("metadata array size overflows");
        }
        cursor.skip(count * size);
        return;
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        skip_value(cursor, element_type, depth + 1);
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        fail("general.alignment must be a non-zero power of two");
    }
    const auto mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        fail("alignment overflows");
    }
    return (value + mask) & ~mask;
}

std::uint64_t tensor_bytes(GgmlType type, std::uint64_t elements) {
    std::uint64_t element_size = 0;
    switch (type) {
    case GgmlType::i8:
        element_size = 1;
        break;
    case GgmlType::f16:
    case GgmlType::i16:
    case GgmlType::bf16:
        element_size = 2;
        break;
    case GgmlType::f32:
    case GgmlType::i32:
        element_size = 4;
        break;
    case GgmlType::i64:
    case GgmlType::f64:
        element_size = 8;
        break;
    default:
        fail("unsupported tensor type " + std::to_string(static_cast<std::uint32_t>(type)));
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_size) {
        fail("tensor byte size overflows");
    }
    return elements * element_size;
}

template <typename T>
T load_scalar(const std::byte *data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

} // namespace

GgufFile::GgufFile(const std::filesystem::path &path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        fail_errno("cannot open", path);
    }

    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
        const auto saved_errno = errno;
        close();
        errno = saved_errno;
        fail_errno("cannot stat", path);
    }
    if (status.st_size < 24) {
        close();
        fail("file is too small");
    }
    file_size_ = static_cast<std::uint64_t>(status.st_size);
    const auto *mapping = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping == MAP_FAILED) {
        mapping_ = nullptr;
        const auto saved_errno = errno;
        close();
        errno = saved_errno;
        fail_errno("cannot mmap", path);
    }
    mapping_ = static_cast<const std::byte *>(mapping);

    try {
        Cursor cursor(mapping_, file_size_);
        if (cursor.read<std::uint32_t>() != gguf_magic) {
            fail("bad magic");
        }
        version_ = cursor.read<std::uint32_t>();
        if (version_ != supported_version) {
            fail("only version 3 is supported");
        }
        const auto tensor_count = cursor.read<std::uint64_t>();
        const auto metadata_count = cursor.read<std::uint64_t>();
        if (tensor_count > cursor.remaining() || metadata_count > cursor.remaining()) {
            fail("directory count exceeds file size");
        }
        metadata_.reserve(static_cast<std::size_t>(metadata_count));
        tensors_.reserve(static_cast<std::size_t>(tensor_count));

        for (std::uint64_t index = 0; index < metadata_count; ++index) {
            const auto key = cursor.read_string();
            const auto type = read_value_type(cursor);
            const auto value_offset = cursor.position();
            if (find_metadata(key) != nullptr) {
                fail("duplicate metadata key '" + std::string(key) + "'");
            }
            metadata_.push_back({key, type, value_offset});
            skip_value(cursor, type);
        }

        if (const auto value = integer("general.alignment")) {
            alignment_ = *value;
        }

        for (std::uint64_t index = 0; index < tensor_count; ++index) {
            GgufTensor tensor;
            tensor.name = cursor.read_string();
            if (find_tensor(tensor.name) != nullptr) {
                fail("duplicate tensor name '" + std::string(tensor.name) + "'");
            }
            const auto dimension_count = cursor.read<std::uint32_t>();
            if (dimension_count == 0 || dimension_count > max_dimensions) {
                fail("tensor has unsupported dimension count");
            }
            tensor.elements = 1;
            tensor.dimensions.reserve(dimension_count);
            for (std::uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
                const auto extent = cursor.read<std::uint64_t>();
                if (extent == 0 ||
                    tensor.elements > std::numeric_limits<std::uint64_t>::max() / extent) {
                    fail("invalid tensor dimensions");
                }
                tensor.elements *= extent;
                tensor.dimensions.push_back(extent);
            }
            tensor.type = static_cast<GgmlType>(cursor.read<std::uint32_t>());
            tensor.offset = cursor.read<std::uint64_t>();
            tensor.bytes = tensor_bytes(tensor.type, tensor.elements);
            tensors_.push_back(std::move(tensor));
        }

        const auto data_offset = align_up(cursor.position(), alignment_);
        if (data_offset > file_size_) {
            fail("tensor data begins outside file");
        }
        for (auto &tensor : tensors_) {
            if (tensor.offset > std::numeric_limits<std::uint64_t>::max() - data_offset) {
                fail("tensor offset overflows");
            }
            tensor.offset += data_offset;
            (void)at(tensor.offset, tensor.bytes);
        }
    } catch (...) {
        close();
        throw;
    }
}

GgufFile::~GgufFile() {
    close();
}

GgufFile::GgufFile(GgufFile &&other) noexcept {
    *this = std::move(other);
}

GgufFile &GgufFile::operator=(GgufFile &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    fd_ = std::exchange(other.fd_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    file_size_ = std::exchange(other.file_size_, 0);
    version_ = std::exchange(other.version_, 0);
    alignment_ = std::exchange(other.alignment_, 32);
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    return *this;
}

const GgufMetadata *GgufFile::find_metadata(std::string_view key) const noexcept {
    for (const auto &entry : metadata_) {
        if (entry.key == key) {
            return &entry;
        }
    }
    return nullptr;
}

const GgufTensor *GgufFile::find_tensor(std::string_view name) const noexcept {
    for (const auto &tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

std::optional<std::uint64_t> GgufFile::integer(std::string_view key) const {
    const auto *entry = find_metadata(key);
    if (entry == nullptr) {
        return std::nullopt;
    }
    const auto *data = at(entry->value_offset, scalar_size(entry->type));
    switch (entry->type) {
    case GgufValueType::u8:
        return load_scalar<std::uint8_t>(data);
    case GgufValueType::u16:
        return load_scalar<std::uint16_t>(data);
    case GgufValueType::u32:
        return load_scalar<std::uint32_t>(data);
    case GgufValueType::u64:
        return load_scalar<std::uint64_t>(data);
    case GgufValueType::i8: {
        const auto value = load_scalar<std::int8_t>(data);
        return value >= 0 ? std::optional<std::uint64_t>(value) : std::nullopt;
    }
    case GgufValueType::i16: {
        const auto value = load_scalar<std::int16_t>(data);
        return value >= 0 ? std::optional<std::uint64_t>(value) : std::nullopt;
    }
    case GgufValueType::i32: {
        const auto value = load_scalar<std::int32_t>(data);
        return value >= 0 ? std::optional<std::uint64_t>(value) : std::nullopt;
    }
    case GgufValueType::i64: {
        const auto value = load_scalar<std::int64_t>(data);
        return value >= 0 ? std::optional<std::uint64_t>(value) : std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

std::optional<double> GgufFile::number(std::string_view key) const {
    const auto *entry = find_metadata(key);
    if (entry == nullptr) {
        return std::nullopt;
    }
    const auto *data = at(entry->value_offset, scalar_size(entry->type));
    if (entry->type == GgufValueType::f32) {
        return load_scalar<float>(data);
    }
    if (entry->type == GgufValueType::f64) {
        return load_scalar<double>(data);
    }
    if (const auto value = integer(key)) {
        return static_cast<double>(*value);
    }
    return std::nullopt;
}

std::optional<bool> GgufFile::boolean(std::string_view key) const {
    const auto *entry = find_metadata(key);
    if (entry == nullptr || entry->type != GgufValueType::boolean) {
        return std::nullopt;
    }
    return load_scalar<std::uint8_t>(at(entry->value_offset, 1)) != 0;
}

std::optional<std::string_view> GgufFile::string(std::string_view key) const {
    const auto *entry = find_metadata(key);
    if (entry == nullptr || entry->type != GgufValueType::string) {
        return std::nullopt;
    }
    Cursor cursor(mapping_, file_size_);
    cursor.skip(entry->value_offset);
    return cursor.read_string();
}

std::span<const std::byte> GgufFile::tensor_data(const GgufTensor &tensor) const {
    return {at(tensor.offset, tensor.bytes), static_cast<std::size_t>(tensor.bytes)};
}

const std::byte *GgufFile::at(std::uint64_t offset, std::uint64_t size) const {
    if (offset > file_size_ || size > file_size_ - offset) {
        fail("range points outside file");
    }
    return mapping_ + offset;
}

void GgufFile::close() noexcept {
    metadata_.clear();
    tensors_.clear();
    if (mapping_ != nullptr) {
        ::munmap(const_cast<std::byte *>(mapping_), file_size_);
        mapping_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
}

std::string_view ggml_type_name(GgmlType type) noexcept {
    switch (type) {
    case GgmlType::f32:
        return "f32";
    case GgmlType::f16:
        return "f16";
    case GgmlType::i8:
        return "i8";
    case GgmlType::i16:
        return "i16";
    case GgmlType::i32:
        return "i32";
    case GgmlType::i64:
        return "i64";
    case GgmlType::f64:
        return "f64";
    case GgmlType::bf16:
        return "bf16";
    }
    return "unknown";
}

} // namespace adi
