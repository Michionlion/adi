#include "adi/model.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

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

void require_optional_string(
    const GgufFile &file,
    std::string_view key,
    std::string_view expected) {
    if (const auto value = file.string(key); value && *value != expected) {
        throw std::runtime_error(
            "model: unsupported metadata '" + std::string(key) + "'");
    }
}

double required_number(const GgufFile &file, std::string_view key) {
    const auto value = file.number(key);
    if (!value || !std::isfinite(*value)) {
        throw std::runtime_error(
            "model: missing or invalid metadata '" + std::string(key) + "'");
    }
    return *value;
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

void require_shape(
    const GgufFile &file,
    std::string_view name,
    GgmlType type,
    std::initializer_list<std::uint64_t> dimensions) {
    const auto &tensor = required_tensor(file, name, type);
    if (!std::equal(
            tensor.dimensions.begin(),
            tensor.dimensions.end(),
            dimensions.begin(),
            dimensions.end())) {
        throw std::runtime_error(
            "model: unexpected tensor shape '" + std::string(name) + "'");
    }
}

void require_finite_bf16(
    std::span<const std::uint16_t> values,
    std::string_view name) {
    if (std::any_of(values.begin(), values.end(), [](std::uint16_t value) {
            return !std::isfinite(bf16_to_f32(value));
        })) {
        throw std::runtime_error(
            "model: nonfinite BF16 tensor '" + std::string(name) + "'");
    }
}

void require_finite_f16(
    std::span<const std::uint16_t> values,
    std::string_view name) {
    if (std::any_of(values.begin(), values.end(), [](std::uint16_t value) {
            return !std::isfinite(f16_to_f32(value));
        })) {
        throw std::runtime_error(
            "model: nonfinite F16 tensor '" + std::string(name) + "'");
    }
}

void require_matrix_shape(
    const Bf16Matrix &matrix,
    std::uint32_t rows,
    std::uint32_t columns,
    std::string_view name) {
    if (matrix.rows != rows || matrix.columns != columns ||
        matrix.values.size() != static_cast<std::size_t>(rows) * columns) {
        throw std::runtime_error(
            "model: unexpected BF16 matrix shape '" + std::string(name) + "'");
    }
    require_finite_bf16(matrix.values, name);
}

void require_vector_shape(
    std::span<const std::uint16_t> vector,
    std::size_t size,
    std::string_view name) {
    if (vector.size() != size) {
        throw std::runtime_error(
            "model: unexpected BF16 vector shape '" + std::string(name) + "'");
    }
    require_finite_bf16(vector, name);
}

void require_ne_shape(
    const MachNeMatrix &matrix,
    std::uint32_t rows,
    std::uint32_t columns,
    std::string_view name) {
    constexpr std::size_t words_per_tile = 64;
    if (matrix.rows != rows || matrix.columns != columns ||
        matrix.trellis.size() !=
            static_cast<std::size_t>(rows / 16) * (columns / 16) *
                words_per_tile ||
        matrix.su.size() != columns || matrix.sv.size() != rows ||
        matrix.tlut.size() != 512 * 2 ||
        !std::isfinite(matrix.weight_scale) || matrix.weight_scale <= 0.0F) {
        throw std::runtime_error(
            "model: unexpected non-expert shape '" + std::string(name) + "'");
    }
    const auto valid_sign = [](std::int8_t value) {
        return value == -1 || value == 1;
    };
    if (!std::all_of(matrix.su.begin(), matrix.su.end(), valid_sign) ||
        !std::all_of(matrix.sv.begin(), matrix.sv.end(), valid_sign)) {
        throw std::runtime_error(
            "model: invalid non-expert sign tensor '" + std::string(name) + "'");
    }
}

} // namespace

MachModel::MachModel(const std::filesystem::path &path) : file_(path) {
    require_string(file_, "general.architecture", "qwen35moe");
    require_string(file_, "adi.model_family", "mach1");
    require_string(file_, "adi.expert_codec", "trained_susv_wave_gamma_chunked_v1");
    require_string(file_, "adi.ne_codec", "canon_rht_bitshift_trellis_intlattice");
    require_string(file_, "adi.embedding_codec", "affine_int4_g64_exceptions");
    require_string(file_, "adi.head_codec", "int5_g64");
    require_string(file_, "tokenizer.ggml.model", "gpt2");
    require_string(file_, "tokenizer.ggml.pre", "qwen2");
    // Format v1 already defines these conventions. New packs spell them out;
    // the optional reads retain compatibility with the original v1 container.
    require_optional_string(
        file_, "adi.regular_rms_norm", "qwen3next_zero_centered");
    require_optional_string(file_, "adi.gated_rms_norm", "multiplicative");
    require_optional_string(
        file_, "adi.tokenizer_normalizer", "nfc_unicode_15_0");
    require_optional_string(
        file_, "adi.tokenizer_pretokenizer", "qwen35_unicode_16_0_regex");
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
    const auto name = file_.string("general.name");
    if (!name || name->empty() || !valid_utf8(*name)) {
        throw std::runtime_error("model: missing general.name");
    }
    name_ = *name;

    config_ = {
        required_u32(file_, "qwen35moe.block_count"),
        required_u32(file_, "qwen35moe.embedding_length"),
        required_u32(file_, "qwen35moe.expert_count"),
        required_u32(file_, "qwen35moe.expert_used_count"),
        required_u32(file_, "qwen35moe.expert_feed_forward_length"),
        required_u32(file_, "qwen35moe.context_length"),
        0,
    };
    if (config_.layers != 40 || config_.hidden != 2048 ||
        config_.experts != 256 || config_.active_experts != 8 ||
        config_.expert_hidden != 512 || config_.context != 262144) {
        throw std::runtime_error("model: unsupported Mach-1 geometry");
    }
    if (required_u32(file_, "qwen35moe.expert_shared_feed_forward_length") != 512 ||
        required_u32(file_, "qwen35moe.attention.head_count") != 16 ||
        required_u32(file_, "qwen35moe.attention.head_count_kv") != 2 ||
        required_u32(file_, "qwen35moe.full_attention_interval") != 4 ||
        required_u32(file_, "qwen35moe.ssm.conv_kernel") != 4 ||
        required_u32(file_, "qwen35moe.ssm.inner_size") != 4096 ||
        required_u32(file_, "qwen35moe.ssm.state_size") != 128 ||
        required_u32(file_, "qwen35moe.ssm.time_step_rank") != 32 ||
        required_u32(file_, "qwen35moe.ssm.group_count") != 16 ||
        std::abs(required_number(
                     file_, "qwen35moe.attention.layer_norm_rms_epsilon") -
                 1.0e-6) >
            1.0e-12 ||
        required_number(file_, "qwen35moe.rope.freq_base") != 10000000.0) {
        throw std::runtime_error("model: unsupported Qwen3.5 geometry");
    }

    (void)required_tensor(file_, "mach.tlut.expert.tlut", GgmlType::f32);
    (void)required_tensor(file_, "mach.tlut.ne.tlut", GgmlType::f32);
    const auto &embedding =
        required_tensor(file_, "mach.embedding.q_packed", GgmlType::i8);
    if (embedding.dimensions.size() != 2 ||
        embedding.dimensions[0] * 2 != config_.hidden ||
        embedding.dimensions[1] > 0xFFFFFFFFULL) {
        throw std::runtime_error("model: malformed embedding geometry");
    }
    config_.vocabulary = static_cast<std::uint32_t>(embedding.dimensions[1]);
    if (config_.vocabulary != 248320) {
        throw std::runtime_error("model: unsupported Mach-1 vocabulary");
    }
    validate_manifest();
}

MachNeMatrix MachModel::non_expert(
    std::uint32_t layer,
    std::string_view source_name) const {
    if (layer >= config_.layers) {
        throw std::out_of_range("model: layer index is out of range");
    }
    const auto prefix = "mach.ne." + std::to_string(layer) + "." +
                        std::string(source_name) + "|";
    const auto &trellis_tensor =
        required_tensor(file_, prefix + "trellis", GgmlType::i16);
    const auto &su_tensor = required_tensor(file_, prefix + "SU", GgmlType::i8);
    const auto &sv_tensor = required_tensor(file_, prefix + "SV", GgmlType::i8);
    const auto &scale_tensor =
        required_tensor(file_, prefix + "Wscale", GgmlType::f32);
    const auto &tlut_tensor =
        required_tensor(file_, "mach.tlut.ne.tlut", GgmlType::f32);
    const auto trellis = view<std::uint16_t>(file_, trellis_tensor);
    const auto su = view<std::int8_t>(file_, su_tensor);
    const auto sv = view<std::int8_t>(file_, sv_tensor);
    const auto scale = view<float>(file_, scale_tensor);
    const auto tlut = view<float>(file_, tlut_tensor);
    if (su.size() > 0xFFFFFFFFULL || sv.size() > 0xFFFFFFFFULL ||
        scale.size() != 1 || su_tensor.dimensions.size() != 1 ||
        sv_tensor.dimensions.size() != 1 ||
        scale_tensor.dimensions != std::vector<std::uint64_t>{1}) {
        throw std::runtime_error("model: malformed non-expert tensor '" + prefix + "'");
    }
    MachNeMatrix result{
        static_cast<std::uint32_t>(sv.size()),
        static_cast<std::uint32_t>(su.size()),
        trellis,
        su,
        sv,
        scale[0],
        tlut,
    };
    const auto expected_trellis =
        static_cast<std::size_t>(result.rows / 16) *
        (result.columns / 16) * 64;
    if (result.rows % 16 != 0 || result.columns % 16 != 0 ||
        result.trellis.size() != expected_trellis ||
        result.tlut.size() != 512 * 2 ||
        trellis_tensor.dimensions !=
            std::vector<std::uint64_t>{64, expected_trellis / 64}) {
        throw std::runtime_error("model: malformed non-expert tensor '" + prefix + "'");
    }
    return result;
}

MachEmbedding MachModel::embedding() const {
    const auto &packed_tensor =
        required_tensor(file_, "mach.embedding.q_packed", GgmlType::i8);
    const auto &minimum_tensor =
        required_tensor(file_, "mach.embedding.mn", GgmlType::f16);
    const auto &maximum_tensor =
        required_tensor(file_, "mach.embedding.mx", GgmlType::f16);
    const auto &index_tensor =
        required_tensor(file_, "mach.embedding.exc_idx", GgmlType::i32);
    const auto &bits_tensor =
        required_tensor(file_, "mach.embedding.exc_bits", GgmlType::i16);
    const auto result = MachEmbedding{
        config_.vocabulary,
        config_.hidden,
        view<std::uint8_t>(file_, packed_tensor),
        view<std::uint16_t>(file_, minimum_tensor),
        view<std::uint16_t>(file_, maximum_tensor),
        view<std::uint32_t>(file_, index_tensor),
        view<std::uint16_t>(file_, bits_tensor),
    };
    const auto groups = static_cast<std::size_t>(config_.hidden / 64);
    const auto elements =
        static_cast<std::uint64_t>(config_.vocabulary) * config_.hidden;
    if (packed_tensor.dimensions !=
            std::vector<std::uint64_t>{
                config_.hidden / 2, config_.vocabulary} ||
        minimum_tensor.dimensions !=
            std::vector<std::uint64_t>{groups, config_.vocabulary} ||
        maximum_tensor.dimensions != minimum_tensor.dimensions ||
        index_tensor.dimensions.size() != 1 ||
        bits_tensor.dimensions != index_tensor.dimensions ||
        result.packed.size() != elements / 2 ||
        result.minimum_f16.size() !=
            static_cast<std::size_t>(config_.vocabulary) * groups ||
        result.maximum_f16.size() != result.minimum_f16.size() ||
        result.exception_indexes.size() != result.exception_bf16.size() ||
        !std::is_sorted(
            result.exception_indexes.begin(), result.exception_indexes.end()) ||
        std::adjacent_find(
            result.exception_indexes.begin(), result.exception_indexes.end()) !=
            result.exception_indexes.end() ||
        (!result.exception_indexes.empty() &&
         result.exception_indexes.back() >= elements)) {
        throw std::runtime_error("model: malformed embedding tensors");
    }
    return result;
}

MachHeadChunk MachModel::head_chunk(std::uint32_t chunk) const {
    constexpr std::uint32_t chunk_count = 8;
    if (chunk >= chunk_count || config_.vocabulary % chunk_count != 0) {
        throw std::out_of_range("model: output chunk index is out of range");
    }
    const auto rows = config_.vocabulary / chunk_count;
    const auto row_begin = chunk * rows;
    const auto row_end = row_begin + rows;
    const auto source = "LMHEADCHUNK:" + std::to_string(row_begin) + ":" +
                        std::to_string(row_end);
    const auto prefix = "mach.output." + std::to_string(chunk) + "." + source + "|";
    const auto &packed_tensor =
        required_tensor(file_, prefix + "qp", GgmlType::i8);
    const auto &scale_tensor =
        required_tensor(file_, prefix + "gscale", GgmlType::f16);

    std::span<const std::uint32_t> protected_rows;
    std::span<const std::uint16_t> protected_bf16;
    const auto *rows_tensor = file_.find_tensor(prefix + "prot_rows");
    const auto *dense_tensor = file_.find_tensor(prefix + "prot_dense");
    if ((rows_tensor == nullptr) != (dense_tensor == nullptr)) {
        throw std::runtime_error("model: incomplete protected output rows");
    }
    if (rows_tensor != nullptr) {
        const auto *tensor = rows_tensor;
        if (tensor->type != GgmlType::i32) {
            throw std::runtime_error("model: invalid protected output row indexes");
        }
        protected_rows = view<std::uint32_t>(file_, *tensor);
        if (dense_tensor->type != GgmlType::bf16) {
            throw std::runtime_error("model: invalid protected output rows");
        }
        protected_bf16 = view<std::uint16_t>(file_, *dense_tensor);
    }
    const auto result = MachHeadChunk{
        rows,
        config_.hidden,
        view<std::uint8_t>(file_, packed_tensor),
        view<std::uint16_t>(file_, scale_tensor),
        protected_rows,
        protected_bf16,
    };
    if (packed_tensor.dimensions !=
            std::vector<std::uint64_t>{
                static_cast<std::uint64_t>(config_.hidden) / 8 * 5, rows} ||
        scale_tensor.dimensions !=
            std::vector<std::uint64_t>{config_.hidden / 64, rows} ||
        (rows_tensor != nullptr &&
         (rows_tensor->dimensions.size() != 1 ||
          dense_tensor->dimensions !=
              std::vector<std::uint64_t>{
                  config_.hidden, rows_tensor->elements})) ||
        result.packed.size() !=
            static_cast<std::size_t>(rows) * config_.hidden / 8 * 5 ||
        result.group_scale_f16.size() !=
            static_cast<std::size_t>(rows) * config_.hidden / 64 ||
        result.protected_bf16.size() !=
            result.protected_rows.size() * config_.hidden ||
        !std::is_sorted(
            result.protected_rows.begin(), result.protected_rows.end()) ||
        std::adjacent_find(
            result.protected_rows.begin(), result.protected_rows.end()) !=
            result.protected_rows.end() ||
        (!result.protected_rows.empty() &&
         result.protected_rows.back() >= rows)) {
        throw std::runtime_error("model: malformed output head chunk");
    }
    return result;
}

Bf16Matrix MachModel::bf16_matrix(std::string_view source_name) const {
    const auto name = "hf." + std::string(source_name);
    const auto &tensor = required_tensor(file_, name, GgmlType::bf16);
    if (tensor.dimensions.size() != 2 ||
        tensor.dimensions[0] > 0xFFFFFFFFULL ||
        tensor.dimensions[1] > 0xFFFFFFFFULL) {
        throw std::runtime_error("model: expected BF16 matrix '" + name + "'");
    }
    return {
        static_cast<std::uint32_t>(tensor.dimensions[1]),
        static_cast<std::uint32_t>(tensor.dimensions[0]),
        view<std::uint16_t>(file_, tensor),
    };
}

std::span<const std::uint16_t> MachModel::bf16_vector(
    std::string_view source_name) const {
    const auto name = "hf." + std::string(source_name);
    const auto &tensor = required_tensor(file_, name, GgmlType::bf16);
    if (tensor.dimensions.size() != 1) {
        throw std::runtime_error("model: expected BF16 vector '" + name + "'");
    }
    return view<std::uint16_t>(file_, tensor);
}

std::span<const std::uint16_t> MachModel::bf16_data(
    std::string_view source_name) const {
    const auto name = "hf." + std::string(source_name);
    const auto &tensor = required_tensor(file_, name, GgmlType::bf16);
    return view<std::uint16_t>(file_, tensor);
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
    if (trellis_tensor.dimensions !=
            std::vector<std::uint64_t>{24, tiles, 32} ||
        su_tensor.dimensions !=
            std::vector<std::uint64_t>{columns, 32} ||
        sv_tensor.dimensions != std::vector<std::uint64_t>{rows, 32} ||
        gamma_tensor.dimensions !=
            std::vector<std::uint64_t>{gamma_values, 32} ||
        trellis_all.size() != trellis_words * 32 ||
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

void MachModel::validate_manifest() const {
    require_shape(
        file_,
        "mach.tlut.expert.tlut",
        GgmlType::f32,
        {8, 32768});
    require_shape(
        file_,
        "mach.tlut.ne.tlut",
        GgmlType::f32,
        {2, 512});
    const auto expert_tlut =
        view<float>(
            file_,
            required_tensor(file_, "mach.tlut.expert.tlut", GgmlType::f32));
    const auto ne_tlut =
        view<float>(
            file_,
            required_tensor(file_, "mach.tlut.ne.tlut", GgmlType::f32));
    if (expert_tlut.size() != 32768 * 8 || ne_tlut.size() != 512 * 2 ||
        std::any_of(expert_tlut.begin(), expert_tlut.end(), [](float value) {
            return !std::isfinite(value);
        }) ||
        std::any_of(ne_tlut.begin(), ne_tlut.end(), [](float value) {
            return !std::isfinite(value);
        })) {
        throw std::runtime_error("model: malformed lookup tables");
    }

    const auto embedding_data = embedding();
    require_finite_f16(embedding_data.minimum_f16, "mach.embedding.mn");
    require_finite_f16(embedding_data.maximum_f16, "mach.embedding.mx");
    require_finite_bf16(
        embedding_data.exception_bf16, "mach.embedding.exc_bits");
    for (std::size_t index = 0; index < embedding_data.minimum_f16.size(); ++index) {
        if (f16_to_f32(embedding_data.minimum_f16[index]) >
            f16_to_f32(embedding_data.maximum_f16[index])) {
            throw std::runtime_error("model: embedding minimum exceeds maximum");
        }
    }

    for (std::uint32_t layer = 0; layer < config_.layers; ++layer) {
        const auto prefix =
            "model.language_model.layers." + std::to_string(layer) + ".";
        require_vector_shape(
            bf16_vector(prefix + "input_layernorm.weight"),
            config_.hidden,
            prefix + "input_layernorm.weight");
        require_vector_shape(
            bf16_vector(prefix + "post_attention_layernorm.weight"),
            config_.hidden,
            prefix + "post_attention_layernorm.weight");
        require_matrix_shape(
            bf16_matrix(prefix + "mlp.gate.weight"),
            config_.experts,
            config_.hidden,
            prefix + "mlp.gate.weight");
        require_matrix_shape(
            bf16_matrix(prefix + "mlp.shared_expert_gate.weight"),
            1,
            config_.hidden,
            prefix + "mlp.shared_expert_gate.weight");

        const auto shared = prefix + "mlp.shared_expert.";
        require_ne_shape(
            non_expert(layer, shared + "gate_proj.weight"),
            config_.expert_hidden,
            config_.hidden,
            shared + "gate_proj.weight");
        require_ne_shape(
            non_expert(layer, shared + "up_proj.weight"),
            config_.expert_hidden,
            config_.hidden,
            shared + "up_proj.weight");
        require_ne_shape(
            non_expert(layer, shared + "down_proj.weight"),
            config_.hidden,
            config_.expert_hidden,
            shared + "down_proj.weight");

        for (std::uint32_t chunk = 0; chunk < config_.experts; chunk += 32) {
            for (const auto projection : {
                     ExpertProjection::gate,
                     ExpertProjection::up,
                     ExpertProjection::down,
                }) {
                const auto matrix = expert(layer, chunk, projection);
                (void)matrix;
                const auto projection_string = projection_name(projection);
                const auto tensor_prefix =
                    "mach.expert." + std::to_string(layer) + ".e" +
                    std::to_string(chunk) + "." + projection_string + ".";
                require_finite_f16(
                    view<std::uint16_t>(
                        file_,
                        required_tensor(
                            file_, tensor_prefix + "su", GgmlType::f16)),
                    tensor_prefix + "su");
                require_finite_f16(
                    view<std::uint16_t>(
                        file_,
                        required_tensor(
                            file_, tensor_prefix + "sv", GgmlType::f16)),
                    tensor_prefix + "sv");
                require_finite_f16(
                    view<std::uint16_t>(
                        file_,
                        required_tensor(
                            file_,
                            tensor_prefix + "wave_gamma",
                            GgmlType::f16)),
                    tensor_prefix + "wave_gamma");
            }
        }

        if ((layer + 1) % 4 == 0) {
            const auto attention = prefix + "self_attn.";
            require_ne_shape(
                non_expert(layer, attention + "q_proj.weight"),
                8192,
                config_.hidden,
                attention + "q_proj.weight");
            require_ne_shape(
                non_expert(layer, attention + "k_proj.weight"),
                512,
                config_.hidden,
                attention + "k_proj.weight");
            require_ne_shape(
                non_expert(layer, attention + "v_proj.weight"),
                512,
                config_.hidden,
                attention + "v_proj.weight");
            require_ne_shape(
                non_expert(layer, attention + "o_proj.weight"),
                config_.hidden,
                4096,
                attention + "o_proj.weight");
            require_vector_shape(
                bf16_vector(attention + "q_norm.weight"),
                256,
                attention + "q_norm.weight");
            require_vector_shape(
                bf16_vector(attention + "k_norm.weight"),
                256,
                attention + "k_norm.weight");
        } else {
            const auto attention = prefix + "linear_attn.";
            require_ne_shape(
                non_expert(layer, attention + "in_proj_qkv.weight"),
                8192,
                config_.hidden,
                attention + "in_proj_qkv.weight");
            require_ne_shape(
                non_expert(layer, attention + "in_proj_z.weight"),
                4096,
                config_.hidden,
                attention + "in_proj_z.weight");
            require_ne_shape(
                non_expert(layer, attention + "out_proj.weight"),
                config_.hidden,
                4096,
                attention + "out_proj.weight");
            require_matrix_shape(
                bf16_matrix(attention + "in_proj_a.weight"),
                32,
                config_.hidden,
                attention + "in_proj_a.weight");
            require_matrix_shape(
                bf16_matrix(attention + "in_proj_b.weight"),
                32,
                config_.hidden,
                attention + "in_proj_b.weight");
            require_vector_shape(
                bf16_vector(attention + "dt_bias"),
                32,
                attention + "dt_bias");
            require_vector_shape(
                bf16_vector(attention + "A_log"),
                32,
                attention + "A_log");
            require_vector_shape(
                bf16_vector(attention + "norm.weight"),
                128,
                attention + "norm.weight");
            require_shape(
                file_,
                "hf." + attention + "conv1d.weight",
                GgmlType::bf16,
                {4, 1, 8192});
            require_finite_bf16(
                bf16_data(attention + "conv1d.weight"),
                attention + "conv1d.weight");
        }
    }
    require_vector_shape(
        bf16_vector("model.language_model.norm.weight"),
        config_.hidden,
        "model.language_model.norm.weight");
    for (std::uint32_t chunk = 0; chunk < 8; ++chunk) {
        const auto head = head_chunk(chunk);
        require_finite_f16(head.group_scale_f16, "output head scale");
        require_finite_bf16(head.protected_bf16, "protected output rows");
    }

    const auto tokens = file_.string_array("tokenizer.ggml.tokens");
    const auto types = file_.integer_array("tokenizer.ggml.token_type");
    const auto merges = file_.string_array("tokenizer.ggml.merges");
    const auto bos = file_.integer("tokenizer.ggml.bos_token_id");
    const auto eos = file_.integer("tokenizer.ggml.eos_token_id");
    if (tokens.size() != config_.vocabulary || types.size() != tokens.size() ||
        merges.empty() || !bos || !eos || *bos >= tokens.size() ||
        *eos >= tokens.size()) {
        throw std::runtime_error("model: malformed tokenizer metadata");
    }
    std::unordered_set<std::string_view> unique_tokens;
    unique_tokens.reserve(tokens.size());
    bool padding_started = false;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].empty() || !valid_utf8(tokens[index]) ||
            types[index] < 1 || types[index] > 5 ||
            !unique_tokens.emplace(tokens[index]).second) {
            throw std::runtime_error("model: invalid tokenizer vocabulary");
        }
        if (types[index] == 5) {
            padding_started = true;
            if (tokens[index] !=
                "[PAD" + std::to_string(index) + "]") {
                throw std::runtime_error("model: invalid tokenizer padding row");
            }
        } else if (padding_started) {
            throw std::runtime_error("model: non-tail tokenizer padding");
        }
    }
    std::unordered_set<std::string_view> unique_merges;
    unique_merges.reserve(merges.size());
    for (const auto merge : merges) {
        const auto separator = merge.find(' ');
        if (!valid_utf8(merge) || separator == std::string_view::npos ||
            separator == 0 ||
            separator + 1 == merge.size() ||
            merge.find(' ', separator + 1) != std::string_view::npos ||
            !unique_merges.emplace(merge).second) {
            throw std::runtime_error("model: invalid tokenizer merges");
        }
        const auto left = merge.substr(0, separator);
        const auto right = merge.substr(separator + 1);
        const auto combined = std::string(left) + std::string(right);
        if (!unique_tokens.contains(left) || !unique_tokens.contains(right) ||
            !unique_tokens.contains(combined)) {
            throw std::runtime_error("model: tokenizer merge references unknown token");
        }
    }
}

} // namespace adi
