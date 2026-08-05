#include "adi/model.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace adi {
namespace {

std::uint32_t required_u32(const GgufFile &file, std::string_view key) {
    const auto value = file.integer(key);
    if (!value || *value > 0xFFFFFFFFULL) {
        throw std::runtime_error("model: missing or invalid metadata '" + std::string(key) + "'");
    }
    return static_cast<std::uint32_t>(*value);
}

void require_string(const GgufFile &file, std::string_view key, std::string_view expected) {
    if (file.string(key) != expected) {
        throw std::runtime_error(
            "model: unsupported metadata '" + std::string(key) + "'");
    }
}

template <typename T>
std::span<const T> view(const GgufFile &file, const GgufTensor &tensor) {
    const auto bytes = file.tensor_data(tensor);
    if (bytes.size() % sizeof(T) != 0 ||
        reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(T) != 0) {
        throw std::runtime_error("model: misaligned tensor '" + std::string(tensor.name) + "'");
    }
    return {
        reinterpret_cast<const T *>(bytes.data()),
        bytes.size() / sizeof(T),
    };
}

const GgufTensor &required_tensor(
    const GgufFile &file,
    std::string_view name,
    GgmlType type) {
    const auto *tensor = file.find_tensor(name);
    if (tensor == nullptr || tensor->type != type) {
        throw std::runtime_error(
            "model: missing or invalid tensor '" + std::string(name) + "'");
    }
    return *tensor;
}

std::string projection_name(ExpertProjection projection) {
    switch (projection) {
    case ExpertProjection::gate:
        return "gate";
    case ExpertProjection::up:
        return "up";
    case ExpertProjection::down:
        return "down";
    }
    throw std::runtime_error("model: invalid expert projection");
}

} // namespace

MachModel::MachModel(const std::filesystem::path &path) : file_(path) {
    require_string(file_, "general.architecture", "qwen35moe");
    require_string(file_, "adi.model_family", "mach1");
    require_string(file_, "adi.expert_codec", "trained_susv_wave_gamma_chunked_v1");
    require_string(file_, "adi.ne_codec", "canon_rht_bitshift_trellis_intlattice");
    require_string(file_, "adi.embedding_codec", "affine_int4_g64_exceptions");
    require_string(file_, "adi.head_codec", "int5_g64");
    if (file_.boolean("adi.additive") != true ||
        required_u32(file_, "adi.format_version") != 1 ||
        required_u32(file_, "adi.expert.k_num") != 3 ||
        required_u32(file_, "adi.expert.k_den") != 2 ||
        required_u32(file_, "adi.expert.l") != 16 ||
        required_u32(file_, "adi.expert.v") != 8 ||
        required_u32(file_, "adi.expert.tlut_bits") != 15 ||
        required_u32(file_, "adi.expert.tile_x") != 16 ||
        required_u32(file_, "adi.expert.tile_y") != 16) {
        throw std::runtime_error("model: unsupported additive format");
    }

    config_ = {
        required_u32(file_, "qwen35moe.block_count"),
        required_u32(file_, "qwen35moe.embedding_length"),
        required_u32(file_, "qwen35moe.expert_count"),
        required_u32(file_, "qwen35moe.expert_used_count"),
        required_u32(file_, "qwen35moe.expert_feed_forward_length"),
        required_u32(file_, "qwen35moe.context_length"),
    };
    if (config_.layers != 40 || config_.hidden != 2048 || config_.experts != 256 ||
        config_.active_experts != 8 || config_.expert_hidden != 512) {
        throw std::runtime_error("model: unsupported Mach-1 geometry");
    }

    (void)required_tensor(file_, "mach.tlut.expert.tlut", GgmlType::f32);
    (void)required_tensor(file_, "mach.tlut.ne.tlut", GgmlType::f32);
    (void)required_tensor(file_, "mach.embedding.q_packed", GgmlType::i8);
}

MachExpertMatrix MachModel::expert(
    std::uint32_t layer,
    std::uint32_t expert_index,
    ExpertProjection projection) const {
    if (layer >= config_.layers || expert_index >= config_.experts) {
        throw std::out_of_range("model: expert index is out of range");
    }
    const auto chunk = (expert_index / 32) * 32;
    const auto offset = expert_index % 32;
    const auto projection_string = projection_name(projection);
    const auto prefix = "mach.expert." + std::to_string(layer) + ".e" +
                        std::to_string(chunk) + "." + projection_string + ".";

    const auto &trellis_tensor =
        required_tensor(file_, prefix + "trellis", GgmlType::i16);
    const auto &su_tensor = required_tensor(file_, prefix + "su", GgmlType::f16);
    const auto &sv_tensor = required_tensor(file_, prefix + "sv", GgmlType::f16);
    const auto &gamma_tensor =
        required_tensor(file_, prefix + "wave_gamma", GgmlType::f16);
    const auto &tlut_tensor =
        required_tensor(file_, "mach.tlut.expert.tlut", GgmlType::f32);

    const auto rows =
        projection == ExpertProjection::down ? config_.hidden : config_.expert_hidden;
    const auto columns =
        projection == ExpertProjection::down ? config_.expert_hidden : config_.hidden;
    const auto tiles = static_cast<std::size_t>(rows / 16) * (columns / 16);
    const auto trellis_words = tiles * 24;
    const auto gamma_values = rows / 16 + columns / 16;
    const auto trellis_all = view<std::uint16_t>(file_, trellis_tensor);
    const auto su_all = view<std::uint16_t>(file_, su_tensor);
    const auto sv_all = view<std::uint16_t>(file_, sv_tensor);
    const auto gamma_all = view<std::uint16_t>(file_, gamma_tensor);
    const auto tlut = view<float>(file_, tlut_tensor);
    if (trellis_all.size() != trellis_words * 32 ||
        su_all.size() != static_cast<std::size_t>(columns) * 32 ||
        sv_all.size() != static_cast<std::size_t>(rows) * 32 ||
        gamma_all.size() != static_cast<std::size_t>(gamma_values) * 32) {
        throw std::runtime_error("model: malformed expert chunk '" + prefix + "'");
    }

    return {
        rows,
        columns,
        trellis_all.subspan(static_cast<std::size_t>(offset) * trellis_words, trellis_words),
        su_all.subspan(static_cast<std::size_t>(offset) * columns, columns),
        sv_all.subspan(static_cast<std::size_t>(offset) * rows, rows),
        gamma_all.subspan(static_cast<std::size_t>(offset) * gamma_values, gamma_values),
        tlut,
    };
}

} // namespace adi
