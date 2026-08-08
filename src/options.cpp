#include "adi/options.hpp"

#include <stdexcept>

namespace adi {

void validate_execution_options(const ExecutionOptions &options) {
    if (options.prefill_ubatch < ExecutionOptions::minimum_prefill_ubatch ||
        options.prefill_ubatch > ExecutionOptions::maximum_prefill_ubatch) {
        throw std::invalid_argument(
            "prefill ubatch must be between 1 and 4096");
    }
}

} // namespace adi
