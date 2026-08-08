#include "adi/server.hpp"

#include "adi/generation.hpp"
#include "adi/json.hpp"
#include "chat.hpp"
#include "server_internal.hpp"
#include "tool_call.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace adi {
namespace {

constexpr std::size_t maximum_headers = 64 * 1024;
constexpr std::size_t maximum_body = 1024 * 1024;
constexpr auto header_deadline = std::chrono::seconds(15);
constexpr auto body_deadline = std::chrono::seconds(30);
constexpr auto send_idle_deadline = std::chrono::seconds(15);
constexpr auto request_deadline = std::chrono::minutes(30);
constexpr std::uint32_t maximum_connections = 64;
std::atomic<std::uint64_t> response_counter = 0;
using TimePoint = std::chrono::steady_clock::time_point;
using server_detail::SocketHandle;

#ifdef _WIN32
using NativeSocket = SOCKET;
constexpr SocketHandle invalid_socket = -1;

NativeSocket native_socket(SocketHandle socket) noexcept {
    return static_cast<NativeSocket>(socket);
}

SocketHandle adi_socket(NativeSocket socket) noexcept {
    return static_cast<SocketHandle>(socket);
}

void close_socket(SocketHandle socket) noexcept {
    ::closesocket(native_socket(socket));
}

int socket_error_code() noexcept {
    return ::WSAGetLastError();
}

bool socket_interrupted(int error) noexcept {
    return error == WSAEINTR;
}

bool socket_would_block(int error) noexcept {
    return error == WSAEWOULDBLOCK;
}

constexpr int winsock_io_size(std::size_t size) noexcept {
    return static_cast<int>(std::min(
        size,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
}

int poll_socket(SocketHandle socket, short events, short &revents, int timeout) noexcept {
    WSAPOLLFD descriptor{native_socket(socket), events, 0};
    const auto result = ::WSAPoll(&descriptor, 1, timeout);
    revents = descriptor.revents;
    return result;
}
#else
using NativeSocket = int;
constexpr SocketHandle invalid_socket = -1;

NativeSocket native_socket(SocketHandle socket) noexcept {
    return static_cast<NativeSocket>(socket);
}

SocketHandle adi_socket(NativeSocket socket) noexcept {
    return static_cast<SocketHandle>(socket);
}

void close_socket(SocketHandle socket) noexcept {
    ::close(native_socket(socket));
}

int socket_error_code() noexcept {
    return errno;
}

bool socket_interrupted(int error) noexcept {
    return error == EINTR;
}

bool socket_would_block(int error) noexcept {
    return error == EAGAIN || error == EWOULDBLOCK;
}

int poll_socket(SocketHandle socket, short events, short &revents, int timeout) noexcept {
    pollfd descriptor{native_socket(socket), events, 0};
    const auto result = ::poll(&descriptor, 1, timeout);
    revents = descriptor.revents;
    return result;
}
#endif

struct Socket {
    SocketHandle descriptor = invalid_socket;
    Socket() = default;
    explicit Socket(SocketHandle value) : descriptor(value) {}
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    Socket(Socket &&other) noexcept
        : descriptor(std::exchange(other.descriptor, invalid_socket)) {}
    Socket &operator=(Socket &&other) noexcept {
        if (this != &other) {
            if (descriptor != invalid_socket) {
                close_socket(descriptor);
            }
            descriptor = std::exchange(other.descriptor, invalid_socket);
        }
        return *this;
    }
    ~Socket() {
        if (descriptor != invalid_socket) {
            close_socket(descriptor);
        }
    }
};

class SocketFailure : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

#ifdef _WIN32
class Winsock {
  public:
    Winsock() {
        WSADATA data{};
        const auto error = ::WSAStartup(MAKEWORD(2, 2), &data);
        if (error != 0) {
            throw SocketFailure(
                "WSAStartup failed: Winsock error " + std::to_string(error));
        }
    }

    ~Winsock() {
        ::WSACleanup();
    }

    Winsock(const Winsock &) = delete;
    Winsock &operator=(const Winsock &) = delete;
};
#endif

struct Connection {
    Connection(SocketHandle socket_value, TimePoint deadline_value)
        : socket(socket_value), deadline(deadline_value) {}

    SocketHandle socket;
    TimePoint deadline;
    std::mutex send_mutex;
};

bool try_acquire_connection(
    std::atomic<std::uint32_t> &active_connections) noexcept {
    auto active = active_connections.load(std::memory_order_relaxed);
    while (active < maximum_connections) {
        if (active_connections.compare_exchange_weak(
                active,
                active + 1,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

using server_detail::ResponseIdentity;
using server_detail::connection_cancelled;
using server_detail::response_json;

[[noreturn]] void socket_error(std::string_view action) {
#ifdef _WIN32
    throw SocketFailure(
        std::string(action) + ": Winsock error " +
        std::to_string(socket_error_code()));
#else
    throw SocketFailure(
        std::string(action) + ": " + std::strerror(errno));
#endif
}

void wait_for_socket(SocketHandle socket, short events, TimePoint deadline) {
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw SocketFailure("socket deadline exceeded");
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto timeout = static_cast<int>(std::clamp<std::int64_t>(
            remaining.count(), 1, std::numeric_limits<int>::max()));
        short revents = 0;
        const auto result = poll_socket(socket, events, revents, timeout);
        if (result > 0) {
            if ((revents & events) != 0) {
                return;
            }
            if ((revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                throw SocketFailure("socket disconnected");
            }
        } else if (result == 0) {
            throw SocketFailure("socket deadline exceeded");
        } else if (!socket_interrupted(socket_error_code())) {
            socket_error("poll");
        }
    }
}

void send_all(Connection &connection, std::string_view data) {
    std::lock_guard lock(connection.send_mutex);
    auto idle = std::min(
        connection.deadline,
        std::chrono::steady_clock::now() + send_idle_deadline);
    while (!data.empty()) {
        wait_for_socket(connection.socket, POLLOUT, idle);
        const auto sent =
#ifdef _WIN32
            ::send(
                native_socket(connection.socket),
                data.data(),
                winsock_io_size(data.size()),
                0);
#else
            ::send(
                native_socket(connection.socket),
                data.data(),
                data.size(),
                MSG_NOSIGNAL | MSG_DONTWAIT);
#endif
        if (sent < 0) {
            const auto error = socket_error_code();
            if (socket_interrupted(error) || socket_would_block(error)) {
                continue;
            }
            socket_error("send");
        }
        if (sent == 0) {
            throw SocketFailure("socket disconnected during send");
        }
        data.remove_prefix(static_cast<std::size_t>(sent));
        idle = std::min(
            connection.deadline,
            std::chrono::steady_clock::now() + send_idle_deadline);
    }
}

std::string lower(std::string_view value) {
    std::string result(value);
    std::transform(
        result.begin(), result.end(), result.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return result;
}

std::size_t content_length(std::string_view headers) {
    std::size_t position = headers.find("\r\n") + 2;
    while (position < headers.size()) {
        const auto end = headers.find("\r\n", position);
        if (end == std::string_view::npos || end == position) {
            break;
        }
        const auto line = headers.substr(position, end - position);
        const auto colon = line.find(':');
        if (colon != std::string_view::npos &&
            lower(line.substr(0, colon)) == "content-length") {
            auto value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') {
                value.remove_prefix(1);
            }
            std::size_t length = 0;
            const auto [parsed, error] =
                std::from_chars(value.data(), value.data() + value.size(), length);
            if (error != std::errc{} || parsed != value.data() + value.size()) {
                throw std::runtime_error("invalid Content-Length");
            }
            return length;
        }
        position = end + 2;
    }
    return 0;
}

std::size_t receive_some(
    SocketHandle socket,
    std::span<char> buffer,
    TimePoint deadline) {
    wait_for_socket(socket, POLLIN, deadline);
    for (;;) {
        const auto received =
#ifdef _WIN32
            ::recv(
                native_socket(socket),
                buffer.data(),
                winsock_io_size(buffer.size()),
                0);
#else
            ::recv(native_socket(socket), buffer.data(), buffer.size(), MSG_DONTWAIT);
#endif
        if (received > 0) {
            return static_cast<std::size_t>(received);
        }
        if (received == 0) {
            throw SocketFailure("client disconnected while sending request");
        }
        const auto error = socket_error_code();
        if (socket_interrupted(error)) {
            continue;
        }
        if (socket_would_block(error)) {
            wait_for_socket(socket, POLLIN, deadline);
            continue;
        }
        socket_error("receive");
    }
}

std::string receive_request(SocketHandle socket, TimePoint overall_deadline) {
    std::string request;
    char buffer[8192];
    std::size_t header_end = std::string::npos;
    const auto headers_end = std::min(
        overall_deadline,
        std::chrono::steady_clock::now() + header_deadline);
    while (header_end == std::string::npos) {
        const auto received = receive_some(socket, buffer, headers_end);
        request.append(buffer, received);
        if (request.size() > maximum_headers) {
            throw std::runtime_error("request headers are too large");
        }
        header_end = request.find("\r\n\r\n");
    }
    const auto length = content_length(
        std::string_view(request).substr(0, header_end + 4));
    if (length > maximum_body) {
        throw std::runtime_error("request body is too large");
    }
    const auto total = header_end + 4 + length;
    const auto body_end = std::min(
        overall_deadline,
        std::chrono::steady_clock::now() + body_deadline);
    while (request.size() < total) {
        const auto received = receive_some(socket, buffer, body_end);
        request.append(buffer, received);
    }
    request.resize(total);
    return request;
}

std::string error_body(std::string_view message) {
    return "{\"error\":{\"message\":" + json_string(message) +
           ",\"type\":\"invalid_request_error\"}}";
}

void send_response(
    Connection &connection,
    int status,
    std::string_view reason,
    std::string body) {
    const auto headers =
        "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    send_all(connection, headers);
    send_all(connection, body);
}

void reject_busy(SocketHandle socket) noexcept {
    Connection connection{
        socket,
        std::chrono::steady_clock::now() + std::chrono::seconds(1),
    };
    try {
        send_response(
            connection,
            503,
            "Service Unavailable",
            error_body("server is at connection capacity"));
    } catch (...) {
    }
}

std::string content_text(const Json &content) {
    if (const auto *text = content.string()) {
        return *text;
    }
    const auto *parts = content.array();
    if (parts == nullptr) {
        throw std::runtime_error("message content must be a string or array");
    }
    std::string result;
    for (const auto &part : *parts) {
        const auto *type_value = part.find("type");
        const auto *text_value = part.find("text");
        if (type_value == nullptr || type_value->string() == nullptr ||
            text_value == nullptr || text_value->string() == nullptr ||
            (*type_value->string() != "input_text" &&
             *type_value->string() != "output_text")) {
            throw std::runtime_error("only text input content is supported");
        }
        result += *text_value->string();
    }
    return result;
}

struct PreparedRequest {
    std::string prompt;
    std::vector<FunctionTool> tools;
    std::vector<Json> response_tools;
    std::string tool_choice;
    bool parallel_tool_calls = false;

    [[nodiscard]] bool tools_enabled() const noexcept {
        return tool_choice == "auto" && !tools.empty();
    }
};

void append_system_message(
    std::vector<ChatMessage> &messages,
    std::string content) {
    if (messages.size() == 1 && messages.front().role == "system") {
        if (!messages.front().content.empty() && !content.empty()) {
            messages.front().content += "\n\n";
        }
        messages.front().content += std::move(content);
        return;
    }
    if (!messages.empty()) {
        throw std::runtime_error("system message must be at the beginning");
    }
    messages.push_back({"system", std::move(content), {}});
}

void append_function_call(
    const Json &item,
    std::vector<ChatMessage> &messages,
    std::unordered_set<std::string> &call_ids) {
    const auto *call_id = item.find("call_id");
    const auto *name = item.find("name");
    const auto *arguments = item.find("arguments");
    if (call_id == nullptr || call_id->string() == nullptr ||
        call_id->string()->empty() || name == nullptr ||
        name->string() == nullptr || name->string()->empty() ||
        arguments == nullptr || arguments->string() == nullptr) {
        throw std::runtime_error(
            "function_call items require string call_id, name, and arguments");
    }
    if (!call_ids.emplace(*call_id->string()).second) {
        throw std::runtime_error("duplicate function call_id in input");
    }
    auto parsed_arguments = parse_json(*arguments->string());
    if (parsed_arguments.object() == nullptr) {
        throw std::runtime_error("function_call arguments must encode an object");
    }
    if (messages.empty() || messages.back().role != "assistant") {
        messages.push_back({"assistant", "", {}});
    }
    messages.back().tool_calls.push_back(
        {*name->string(), std::move(parsed_arguments)});
}

void append_function_output(
    const Json &item,
    std::vector<ChatMessage> &messages,
    const std::unordered_set<std::string> &call_ids) {
    const auto *call_id = item.find("call_id");
    const auto *output = item.find("output");
    if (call_id == nullptr || call_id->string() == nullptr ||
        output == nullptr) {
        throw std::runtime_error(
            "function_call_output items require string call_id and output");
    }
    if (!call_ids.contains(*call_id->string())) {
        throw std::runtime_error(
            "function_call_output refers to an unknown call_id");
    }
    messages.push_back({"tool", content_text(*output), {}});
}

std::vector<ChatMessage> response_messages(const Json &root) {
    const auto *input = root.find("input");
    if (input == nullptr) {
        throw std::runtime_error("'input' is required");
    }
    std::vector<ChatMessage> rendered_messages;
    if (const auto *instructions = root.find("instructions");
        instructions != nullptr && !instructions->is_null()) {
        if (instructions->string() == nullptr) {
            throw std::runtime_error("'instructions' must be a string");
        }
        append_system_message(rendered_messages, *instructions->string());
    }
    if (const auto *text = input->string()) {
        rendered_messages.push_back({"user", *text, {}});
        return rendered_messages;
    }
    const auto *items = input->array();
    if (items == nullptr || items->empty()) {
        throw std::runtime_error("'input' must be a string or non-empty message array");
    }
    rendered_messages.reserve(rendered_messages.size() + items->size());
    std::unordered_set<std::string> call_ids;
    for (const auto &item : *items) {
        if (item.object() == nullptr) {
            throw std::runtime_error("each input item must be an object");
        }
        const auto *type = item.find("type");
        if (type != nullptr && type->string() == nullptr) {
            throw std::runtime_error("input item 'type' must be a string");
        }
        if (type != nullptr && *type->string() == "function_call") {
            append_function_call(item, rendered_messages, call_ids);
            continue;
        }
        if (type != nullptr && *type->string() == "function_call_output") {
            append_function_output(item, rendered_messages, call_ids);
            continue;
        }
        if (type != nullptr && *type->string() != "message") {
            throw std::runtime_error("unsupported Responses input item type");
        }

        const auto *role = item.find("role");
        const auto *content = item.find("content");
        if (role == nullptr || role->string() == nullptr || content == nullptr) {
            throw std::runtime_error("each input message needs role and content");
        }
        if (*role->string() == "system" || *role->string() == "developer") {
            append_system_message(rendered_messages, content_text(*content));
        } else if (*role->string() == "user" ||
                   *role->string() == "assistant") {
            rendered_messages.push_back(
                {*role->string(), content_text(*content), {}});
        } else {
            throw std::runtime_error("unsupported input message role");
        }
    }
    return rendered_messages;
}

PreparedRequest prepare_request(const Json &root) {
    if (const auto *previous = root.find("previous_response_id");
        previous != nullptr && !previous->is_null()) {
        throw std::runtime_error(
            "previous_response_id is not supported; send the full input history");
    }

    PreparedRequest result;
    if (const auto *tools = root.find("tools");
        tools != nullptr && !tools->is_null()) {
        result.tools = parse_function_tools(*tools);
    }
    result.response_tools.reserve(result.tools.size());
    std::vector<Json> prompt_tools;
    prompt_tools.reserve(result.tools.size());
    for (const auto &tool : result.tools) {
        result.response_tools.push_back(tool.response_definition);
        prompt_tools.push_back(tool.prompt_definition);
    }

    result.tool_choice = result.tools.empty() ? "none" : "auto";
    if (const auto *choice = root.find("tool_choice"); choice != nullptr) {
        if (choice->string() == nullptr ||
            (*choice->string() != "auto" && *choice->string() != "none")) {
            throw std::runtime_error(
                "'tool_choice' must be 'auto' or 'none'");
        }
        result.tool_choice = *choice->string();
    }
    result.parallel_tool_calls = !result.tools.empty();
    if (const auto *parallel = root.find("parallel_tool_calls");
        parallel != nullptr) {
        if (parallel->boolean() == nullptr) {
            throw std::runtime_error("'parallel_tool_calls' must be a boolean");
        }
        result.parallel_tool_calls = *parallel->boolean();
    }

    const auto messages = response_messages(root);
    result.prompt = qwen35_chat_prompt(
        messages,
        result.tools_enabled() ? std::span<const Json>(prompt_tools)
                               : std::span<const Json>{});
    return result;
}

std::uint32_t integer_option(
    const Json &root,
    std::string_view key,
    std::uint32_t fallback) {
    const auto *value = root.find(key);
    if (value == nullptr) {
        return fallback;
    }
    const auto *number = value->number();
    if (number == nullptr || *number < 0.0 ||
        *number > std::numeric_limits<std::uint32_t>::max() ||
        std::floor(*number) != *number) {
        throw std::runtime_error("'" + std::string(key) + "' must be an integer");
    }
    return static_cast<std::uint32_t>(*number);
}

std::optional<std::uint32_t> optional_u32_option(
    const Json &root,
    std::string_view key) {
    const auto *value = root.find(key);
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto *number = value->number();
    if (number == nullptr || *number <= 0.0 ||
        *number > std::numeric_limits<std::uint32_t>::max() ||
        std::floor(*number) != *number) {
        throw std::runtime_error(
            "'" + std::string(key) + "' must be a positive integer");
    }
    return static_cast<std::uint32_t>(*number);
}

float float_option(
    const Json &root,
    std::string_view key,
    float fallback) {
    const auto *value = root.find(key);
    if (value == nullptr) {
        return fallback;
    }
    const auto *number = value->number();
    if (number == nullptr) {
        throw std::runtime_error("'" + std::string(key) + "' must be a number");
    }
    const auto converted = static_cast<float>(*number);
    if (!std::isfinite(converted) ||
        (*number != 0.0 && converted == 0.0F)) {
        throw std::runtime_error("'" + std::string(key) +
                                 "' is outside the supported range");
    }
    return converted;
}

} // namespace

namespace server_detail {

std::string error_event_json(
    std::string_view message,
    std::uint32_t sequence_number) {
    return "{\"type\":\"error\",\"sequence_number\":" +
           std::to_string(sequence_number) +
           ",\"message\":" + json_string(message) + "}";
}

std::string response_json(
    std::string_view model_name,
    const GenerationResult &result,
    const ResponseIdentity &identity) {
    return response_json(
        model_name,
        result,
        identity,
        ParsedModelOutput{result.text, {}},
        ResponseConfiguration{});
}

std::string response_json(
    std::string_view model_name,
    const GenerationResult &result,
    const ResponseIdentity &identity,
    const ParsedModelOutput &output,
    const ResponseConfiguration &configuration) {
    const bool incomplete = result.finish_reason == FinishReason::length;
    const std::string_view response_status =
        incomplete ? "incomplete" : "completed";
    const std::string_view item_status =
        incomplete ? "incomplete" : "completed";
    const std::string_view incomplete_details =
        incomplete ? "{\"reason\":\"max_output_tokens\"}" : "null";
    const double prompt_ms = result.prefill_seconds * 1000.0;
    const double predicted_ms = result.decode_seconds * 1000.0;
    const double prompt_per_token_ms =
        result.input_tokens == 0 ? 0.0 : prompt_ms / result.input_tokens;
    const double predicted_per_token_ms =
        result.output_tokens == 0 ? 0.0 : predicted_ms / result.output_tokens;
    const double prompt_per_second =
        result.prefill_seconds <= 0.0
            ? 0.0
            : result.input_tokens / result.prefill_seconds;
    const double predicted_per_second =
        result.decode_seconds <= 0.0
            ? 0.0
            : result.output_tokens / result.decode_seconds;

    std::string tools_json = "[";
    for (std::size_t index = 0; index < configuration.tools.size(); ++index) {
        if (index != 0) {
            tools_json.push_back(',');
        }
        tools_json += json_dump(configuration.tools[index]);
    }
    tools_json.push_back(']');

    std::string output_json = "[";
    std::size_t output_index = 0;
    if (!output.text.empty() || output.function_calls.empty()) {
        output_json += "{\"id\":" + json_string(identity.message_id) +
                       ",\"type\":\"message\",\"status\":" +
                       json_string(item_status) +
                       ",\"role\":\"assistant\",\"content\":[{\"type\":"
                       "\"output_text\",\"text\":" + json_string(output.text) +
                       ",\"annotations\":[]}]}";
        ++output_index;
    }
    for (std::size_t call_index = 0;
         call_index < output.function_calls.size();
         ++call_index, ++output_index) {
        if (output_index != 0) {
            output_json.push_back(',');
        }
        const auto suffix = identity.id.substr(5) + "_" +
                            std::to_string(call_index);
        const auto &call = output.function_calls[call_index];
        output_json +=
            "{\"id\":" + json_string("fc_" + suffix) +
            ",\"call_id\":" + json_string("call_" + suffix) +
            ",\"type\":\"function_call\",\"name\":" +
            json_string(call.name) + ",\"arguments\":" +
            json_string(call.arguments_json) +
            ",\"status\":\"completed\"}";
    }
    output_json.push_back(']');

    return "{\"id\":" + json_string(identity.id) +
           ",\"object\":\"response\",\"created_at\":" +
           std::to_string(identity.created_at) +
           ",\"status\":" + json_string(response_status) +
           ",\"error\":null,\"incomplete_details\":" +
           std::string(incomplete_details) +
           ",\"model\":" + json_string(model_name) +
           ",\"parallel_tool_calls\":" +
           (configuration.parallel_tool_calls ? "true" : "false") +
           ",\"tool_choice\":" + json_string(configuration.tool_choice) +
           ",\"tools\":" + tools_json + ",\"output\":" + output_json +
           ",\"usage\":{\"input_tokens\":" +
           std::to_string(result.input_tokens) +
           ",\"input_tokens_details\":{\"cache_write_tokens\":0,"
           "\"cached_tokens\":0},\"output_tokens\":" +
           std::to_string(result.output_tokens) +
           ",\"output_tokens_details\":{\"reasoning_tokens\":0},"
           "\"total_tokens\":" +
           std::to_string(result.input_tokens + result.output_tokens) + "},"
           "\"timings\":{\"prompt_n\":" +
           std::to_string(result.input_tokens) +
           ",\"prompt_ms\":" + std::to_string(prompt_ms) +
           ",\"prompt_per_token_ms\":" +
           std::to_string(prompt_per_token_ms) +
           ",\"prompt_per_second\":" +
           std::to_string(prompt_per_second) +
           ",\"predicted_n\":" + std::to_string(result.output_tokens) +
           ",\"predicted_ms\":" + std::to_string(predicted_ms) +
           ",\"predicted_per_token_ms\":" +
           std::to_string(predicted_per_token_ms) +
           ",\"predicted_per_second\":" +
           std::to_string(predicted_per_second) + "}}";
}

std::uint16_t bound_port(SocketHandle socket) {
    sockaddr_in bound_address{};
#ifdef _WIN32
    int address_length = static_cast<int>(sizeof(bound_address));
#else
    socklen_t address_length = sizeof(bound_address);
#endif
    if (::getsockname(
            native_socket(socket),
            reinterpret_cast<sockaddr *>(&bound_address),
            &address_length) != 0) {
        socket_error("getsockname");
    }
    return ntohs(bound_address.sin_port);
}

bool connection_cancelled(
    SocketHandle socket,
    std::chrono::steady_clock::time_point deadline) noexcept {
    if (std::chrono::steady_clock::now() >= deadline) {
        return true;
    }
    short revents = 0;
    const auto ready = poll_socket(
        socket,
#ifdef _WIN32
        POLLIN,
#else
        static_cast<short>(POLLIN | POLLERR | POLLHUP),
#endif
        revents,
        0);
    if (ready < 0) {
        return !socket_interrupted(socket_error_code());
    }
    if (ready == 0) {
        return false;
    }
    if ((revents & (POLLERR | POLLNVAL)) != 0) {
        return true;
    }
    if ((revents & (POLLIN | POLLHUP)) != 0) {
        char byte = 0;
#ifdef _WIN32
        const auto received =
            ::recv(native_socket(socket), &byte, 1, MSG_PEEK);
#else
        const auto received =
            ::recv(native_socket(socket), &byte, 1, MSG_PEEK | MSG_DONTWAIT);
#endif
        // Read EOF includes a valid peer shutdown(SHUT_WR). A failed response
        // send will detect whether the peer also stopped receiving.
        if (received >= 0) {
            return false;
        }
        const auto error = socket_error_code();
        return !socket_would_block(error) && !socket_interrupted(error);
    }
    return false;
}

} // namespace server_detail

namespace {

ResponseIdentity make_identity() {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    const auto sequence = response_counter.fetch_add(1);
    return {
        "resp_" + std::to_string(now) + "_" + std::to_string(sequence),
        "msg_" + std::to_string(now) + "_" + std::to_string(sequence),
        now,
    };
}

GenerationOptions request_options(const Json &request) {
    GenerationOptions options;
    options.max_output_tokens = optional_u32_option(request, "max_output_tokens");
    options.temperature = float_option(request, "temperature", options.temperature);
    options.top_p = float_option(request, "top_p", options.top_p);
    options.seed = integer_option(request, "seed", 0);
    return options;
}

std::string_view request_model(
    const Json &request,
    const MachModel &model) {
    if (const auto *requested = request.find("model"); requested != nullptr) {
        if (requested->string() == nullptr) {
            throw std::runtime_error("'model' must be a string");
        }
        if (*requested->string() != model.name()) {
            throw std::runtime_error(
                "requested model does not match the loaded model");
        }
    }
    return model.name();
}

server_detail::ResponseConfiguration response_configuration(
    const PreparedRequest &request) {
    return {
        request.parallel_tool_calls,
        request.tool_choice,
        request.response_tools,
    };
}

void send_event(
    Connection &connection,
    std::string_view event,
    std::string data) {
    send_all(
        connection,
        "event: " + std::string(event) + "\ndata: " + data + "\n\n");
}

std::string function_suffix(
    const ResponseIdentity &identity,
    std::size_t call_index) {
    return identity.id.substr(5) + "_" + std::to_string(call_index);
}

void emit_message_start(
    Connection &connection,
    const ResponseIdentity &identity,
    std::size_t output_index,
    std::uint32_t &sequence) {
    send_event(
        connection,
        "response.output_item.added",
        "{\"type\":\"response.output_item.added\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"output_index\":" +
            std::to_string(output_index) + ",\"item\":{\"id\":" +
            json_string(identity.message_id) +
            ",\"type\":\"message\",\"status\":\"in_progress\","
            "\"role\":\"assistant\",\"content\":[]}}");
    send_event(
        connection,
        "response.content_part.added",
        "{\"type\":\"response.content_part.added\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"item_id\":" +
            json_string(identity.message_id) + ",\"output_index\":" +
            std::to_string(output_index) +
            ",\"content_index\":0,\"part\":{\"type\":\"output_text\","
            "\"text\":\"\",\"annotations\":[]}}");
}

void emit_text_delta(
    Connection &connection,
    const ResponseIdentity &identity,
    std::size_t output_index,
    std::string_view delta,
    std::uint32_t &sequence) {
    if (delta.empty()) {
        return;
    }
    send_event(
        connection,
        "response.output_text.delta",
        "{\"type\":\"response.output_text.delta\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"item_id\":" +
            json_string(identity.message_id) + ",\"output_index\":" +
            std::to_string(output_index) +
            ",\"content_index\":0,\"delta\":" + json_string(delta) + "}");
}

void emit_message_done(
    Connection &connection,
    const ResponseIdentity &identity,
    std::size_t output_index,
    std::string_view text,
    std::string_view status,
    std::uint32_t &sequence) {
    send_event(
        connection,
        "response.output_text.done",
        "{\"type\":\"response.output_text.done\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"item_id\":" +
            json_string(identity.message_id) + ",\"output_index\":" +
            std::to_string(output_index) +
            ",\"content_index\":0,\"text\":" + json_string(text) + "}");
    send_event(
        connection,
        "response.content_part.done",
        "{\"type\":\"response.content_part.done\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"item_id\":" +
            json_string(identity.message_id) + ",\"output_index\":" +
            std::to_string(output_index) +
            ",\"content_index\":0,\"part\":{\"type\":\"output_text\","
            "\"text\":" + json_string(text) + ",\"annotations\":[]}}");
    send_event(
        connection,
        "response.output_item.done",
        "{\"type\":\"response.output_item.done\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"output_index\":" +
            std::to_string(output_index) + ",\"item\":{\"id\":" +
            json_string(identity.message_id) +
            ",\"type\":\"message\",\"status\":" + json_string(status) +
            ",\"role\":\"assistant\",\"content\":[{\"type\":"
            "\"output_text\",\"text\":" + json_string(text) +
            ",\"annotations\":[]}]}}");
}

void emit_function_call(
    Connection &connection,
    const ResponseIdentity &identity,
    const FunctionCall &call,
    std::size_t output_index,
    std::size_t call_index,
    std::uint32_t &sequence) {
    const auto suffix = function_suffix(identity, call_index);
    const auto item_id = "fc_" + suffix;
    const auto call_id = "call_" + suffix;
    send_event(
        connection,
        "response.output_item.added",
        "{\"type\":\"response.output_item.added\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"output_index\":" +
            std::to_string(output_index) + ",\"item\":{\"id\":" +
            json_string(item_id) + ",\"call_id\":" + json_string(call_id) +
            ",\"type\":\"function_call\",\"name\":" +
            json_string(call.name) +
            ",\"arguments\":\"\",\"status\":\"in_progress\"}}");
    send_event(
        connection,
        "response.function_call_arguments.delta",
        "{\"type\":\"response.function_call_arguments.delta\","
        "\"sequence_number\":" + std::to_string(sequence++) +
            ",\"item_id\":" + json_string(item_id) +
            ",\"output_index\":" + std::to_string(output_index) +
            ",\"delta\":" + json_string(call.arguments_json) + "}");
    send_event(
        connection,
        "response.function_call_arguments.done",
        "{\"type\":\"response.function_call_arguments.done\","
        "\"sequence_number\":" + std::to_string(sequence++) +
            ",\"item_id\":" + json_string(item_id) +
            ",\"output_index\":" + std::to_string(output_index) +
            ",\"arguments\":" + json_string(call.arguments_json) + "}");
    send_event(
        connection,
        "response.output_item.done",
        "{\"type\":\"response.output_item.done\",\"sequence_number\":" +
            std::to_string(sequence++) + ",\"output_index\":" +
            std::to_string(output_index) + ",\"item\":{\"id\":" +
            json_string(item_id) + ",\"call_id\":" + json_string(call_id) +
            ",\"type\":\"function_call\",\"name\":" +
            json_string(call.name) + ",\"arguments\":" +
            json_string(call.arguments_json) +
            ",\"status\":\"completed\"}}");
}

void stream_responses(
    Connection &connection,
    const Json &request,
    const PreparedRequest &prepared,
    const MachModel &model,
    ContinuousBatcher &batcher) {
    const auto headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    send_all(connection, headers);
    std::uint32_t sequence = 0;
    try {
        const auto identity = make_identity();
        const std::string model_name(request_model(request, model));
        const auto configuration = response_configuration(prepared);
        std::string tools_json = "[";
        for (std::size_t index = 0; index < prepared.response_tools.size(); ++index) {
            if (index != 0) {
                tools_json.push_back(',');
            }
            tools_json += json_dump(prepared.response_tools[index]);
        }
        tools_json.push_back(']');
        const auto response_stub =
            "{\"id\":" + json_string(identity.id) +
            ",\"object\":\"response\",\"created_at\":" +
            std::to_string(identity.created_at) +
            ",\"status\":\"in_progress\",\"error\":null,"
            "\"incomplete_details\":null,\"model\":" +
            json_string(model_name) +
            ",\"parallel_tool_calls\":" +
            (prepared.parallel_tool_calls ? "true" : "false") +
            ",\"tool_choice\":" + json_string(prepared.tool_choice) +
            ",\"tools\":" + tools_json + ",\"output\":[]}";
        send_event(
            connection,
            "response.created",
            "{\"type\":\"response.created\",\"sequence_number\":" +
                std::to_string(sequence++) + ",\"response\":" + response_stub + "}");
        send_event(
            connection,
            "response.in_progress",
            "{\"type\":\"response.in_progress\",\"sequence_number\":" +
                std::to_string(sequence++) + ",\"response\":" + response_stub + "}");
        bool message_started = !prepared.tools_enabled();
        bool tool_marker_seen = false;
        std::string streamed_text;
        if (message_started) {
            emit_message_start(connection, identity, 0, sequence);
        }

        std::string pending;
        const auto emit_delta = [&](std::string_view delta) {
            emit_text_delta(connection, identity, 0, delta, sequence);
        };
        const auto drain_pending = [&](bool final) {
            while (!pending.empty()) {
                const auto scan = scan_utf8(pending);
                if (scan.valid_prefix != 0) {
                    emit_delta(
                        std::string_view(pending).substr(0, scan.valid_prefix));
                    pending.erase(0, scan.valid_prefix);
                }
                if (scan.status == Utf8Status::valid) {
                    break;
                }
                if (scan.status == Utf8Status::incomplete) {
                    if (final) {
                        emit_delta("\xEF\xBF\xBD");
                        pending.clear();
                    }
                    break;
                }
                emit_delta("\xEF\xBF\xBD");
                pending.erase(0, scan.error_length);
            }
        };
        auto next_connection_probe =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        const auto cancellation_check = [&]() mutable {
            const auto now = std::chrono::steady_clock::now();
            if (now >= connection.deadline) {
                throw std::runtime_error("request deadline exceeded");
            }
            if (connection_cancelled(
                    connection.socket, connection.deadline)) {
                throw SocketFailure("client disconnected during generation");
            }
            if (now >= next_connection_probe) {
                // A read EOF can be a valid HTTP half-close, so it cannot by
                // itself distinguish a waiting client from a fully closed
                // connection. An SSE comment safely probes the response side.
                send_all(connection, ": keepalive\n\n");
                next_connection_probe = now + std::chrono::milliseconds(250);
            }
            return false;
        };
        const auto result = batcher.generate_from_prompt(
            prepared.prompt,
            request_options(request),
            prepared.tools_enabled()
                ? TokenCallback{[&](std::string_view piece) {
                      if (tool_marker_seen) {
                          return;
                      }
                      // The supported tokenizer represents <tool_call> as one
                      // special token, so syntax never has to be exposed as a
                      // partial text delta.
                      const auto marker = piece.find("<tool_call>");
                      const auto visible = piece.substr(0, marker);
                      if (!visible.empty()) {
                          if (!message_started) {
                              emit_message_start(
                                  connection, identity, 0, sequence);
                              message_started = true;
                          }
                          streamed_text.append(visible);
                          pending.append(visible);
                          drain_pending(false);
                      }
                      tool_marker_seen = marker != std::string_view::npos;
                  }}
                : TokenCallback{[&](std::string_view piece) {
                      pending.append(piece);
                      drain_pending(false);
                  }},
            cancellation_check);
        if (!pending.empty()) {
            drain_pending(true);
        }
        const auto output = prepared.tools_enabled()
                                ? parse_tool_calls(
                                      result.text,
                                      prepared.tools,
                                      prepared.parallel_tool_calls)
                                : ParsedModelOutput{result.text, {}};
        const auto item_status = result.finish_reason == FinishReason::length
                                     ? "incomplete"
                                     : "completed";
        std::size_t output_index = 0;
        if (!output.text.empty() || output.function_calls.empty()) {
            if (!message_started) {
                emit_message_start(connection, identity, output_index, sequence);
                message_started = true;
            }
            if (prepared.tools_enabled()) {
                if (!output.text.starts_with(streamed_text)) {
                    throw std::runtime_error(
                        "streamed text does not match parsed model output");
                }
                emit_text_delta(
                    connection,
                    identity,
                    output_index,
                    std::string_view(output.text).substr(streamed_text.size()),
                    sequence);
            }
            emit_message_done(
                connection,
                identity,
                output_index,
                output.text,
                item_status,
                sequence);
            ++output_index;
        }
        for (std::size_t call_index = 0;
             call_index < output.function_calls.size();
             ++call_index, ++output_index) {
            emit_function_call(
                connection,
                identity,
                output.function_calls[call_index],
                output_index,
                call_index,
                sequence);
        }
        const std::string terminal_event =
            result.finish_reason == FinishReason::length
                ? "response.incomplete"
                : "response.completed";
        send_event(
            connection,
            terminal_event,
            "{\"type\":" + json_string(terminal_event) +
                ",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"response\":" +
                response_json(
                    model_name, result, identity, output, configuration) + "}");
    } catch (const SocketFailure &) {
        return;
    } catch (const std::exception &error) {
        try {
            send_event(
                connection,
                "error",
                server_detail::error_event_json(error.what(), sequence++));
        } catch (const SocketFailure &) {
        }
    }
}

std::string handle_responses(
    const Json &request,
    const PreparedRequest &prepared,
    const MachModel &model,
    ContinuousBatcher &batcher,
    Connection &connection) {
    const auto model_name = request_model(request, model);
    const auto cancellation_check = [&]() {
        if (std::chrono::steady_clock::now() >= connection.deadline) {
            throw std::runtime_error("request deadline exceeded");
        }
        if (connection_cancelled(
                connection.socket, connection.deadline)) {
            throw SocketFailure("client disconnected during generation");
        }
        return false;
    };
    const auto result = batcher.generate_from_prompt(
        prepared.prompt,
        request_options(request),
        {},
        cancellation_check);
    const auto output = prepared.tools_enabled()
                            ? parse_tool_calls(
                                  result.text,
                                  prepared.tools,
                                  prepared.parallel_tool_calls)
                            : ParsedModelOutput{result.text, {}};
    const auto identity = make_identity();
    return response_json(
        model_name,
        result,
        identity,
        output,
        response_configuration(prepared));
}

void handle_client(
    SocketHandle socket,
    const MachModel &model,
    ContinuousBatcher &batcher) noexcept {
    Connection connection{
        socket,
        std::chrono::steady_clock::now() + request_deadline,
    };
    try {
        const auto request = receive_request(socket, connection.deadline);
        const auto line_end = request.find("\r\n");
        const auto request_line = std::string_view(request).substr(0, line_end);
        const auto first_space = request_line.find(' ');
        const auto second_space = request_line.find(' ', first_space + 1);
        if (first_space == std::string_view::npos ||
            second_space == std::string_view::npos) {
            throw std::runtime_error("invalid HTTP request line");
        }
        const auto method = request_line.substr(0, first_space);
        const auto target =
            request_line.substr(first_space + 1, second_space - first_space - 1);
        if (method != "POST" || target != "/v1/responses") {
            send_response(
                connection, 404, "Not Found", error_body("route not found"));
            return;
        }
        const auto body_position = request.find("\r\n\r\n") + 4;
        const auto body = std::string_view(request).substr(body_position);
        const auto parsed = parse_json(body);
        if (parsed.object() == nullptr) {
            throw std::runtime_error("request body must be a JSON object");
        }
        if (const auto *stream = parsed.find("stream");
            stream != nullptr && stream->boolean() == nullptr) {
            throw std::runtime_error("'stream' must be a boolean");
        }
        (void)request_model(parsed, model);
        (void)request_options(parsed);
        const auto prepared = prepare_request(parsed);
        if (const auto *stream = parsed.find("stream");
            stream != nullptr && stream->boolean() != nullptr && *stream->boolean()) {
            stream_responses(connection, parsed, prepared, model, batcher);
        } else {
            send_response(
                connection,
                200,
                "OK",
                handle_responses(
                    parsed, prepared, model, batcher, connection));
        }
    } catch (const SocketFailure &) {
        return;
    } catch (const std::exception &error) {
        try {
            send_response(
                connection, 400, "Bad Request", error_body(error.what()));
        } catch (const SocketFailure &) {
        }
    } catch (...) {
        std::cerr << "adi: unknown client error\n";
    }
}

} // namespace

[[noreturn]] void serve(const ServerOptions &options) {
#ifdef _WIN32
    [[maybe_unused]] Winsock winsock;
#endif
    const MachModel model(options.model);
    Tokenizer tokenizer(model);
    ContinuousBatcher batcher(model, tokenizer, options.execution);
    std::atomic<std::uint32_t> active_connections = 0;
    const auto raw_listener = ::socket(
        AF_INET,
        SOCK_STREAM
#ifndef _WIN32
        | SOCK_CLOEXEC
#endif
        ,
        0);
    Socket listener{adi_socket(raw_listener)};
    if (listener.descriptor == invalid_socket) {
        socket_error("socket");
    }
    int address_option = 1;
    if (::setsockopt(
            native_socket(listener.descriptor),
            SOL_SOCKET,
#ifdef _WIN32
            SO_EXCLUSIVEADDRUSE,
#else
            SO_REUSEADDR,
#endif
            reinterpret_cast<const char *>(&address_option),
            static_cast<int>(sizeof(address_option))) != 0) {
        socket_error("setsockopt");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    if (::inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("server: host must be an IPv4 address");
    }
    if (::bind(
            native_socket(listener.descriptor),
            reinterpret_cast<const sockaddr *>(&address),
            sizeof(address)) != 0) {
        socket_error("bind");
    }
    if (::listen(
            native_socket(listener.descriptor),
            static_cast<int>(maximum_connections)) != 0) {
        socket_error("listen");
    }
    const auto listening_port =
        server_detail::bound_port(listener.descriptor);
    std::cout << "adi: listening on http://" << options.host << ':'
              << listening_port
              << "/v1/responses\n";
    std::cout.flush();
    for (;;) {
#ifdef _WIN32
        const auto raw_client = ::accept(
            native_socket(listener.descriptor), nullptr, nullptr);
#else
        const auto raw_client = ::accept4(
            native_socket(listener.descriptor), nullptr, nullptr, SOCK_CLOEXEC);
#endif
        Socket client{adi_socket(raw_client)};
        if (client.descriptor == invalid_socket) {
            if (socket_interrupted(socket_error_code())) {
                continue;
            }
            socket_error("accept");
        }
#ifdef _WIN32
        u_long nonblocking = 1;
        if (::ioctlsocket(
                native_socket(client.descriptor), FIONBIO, &nonblocking) != 0) {
            std::cerr << "adi: cannot configure client socket: socket error "
                      << socket_error_code() << '\n';
            continue;
        }
#endif
#ifdef _WIN32
        const DWORD io_timeout = static_cast<DWORD>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                send_idle_deadline).count());
        if (::setsockopt(
                native_socket(client.descriptor),
                SOL_SOCKET,
                SO_RCVTIMEO,
                reinterpret_cast<const char *>(&io_timeout),
                static_cast<int>(sizeof(io_timeout))) != 0 ||
            ::setsockopt(
                native_socket(client.descriptor),
                SOL_SOCKET,
                SO_SNDTIMEO,
                reinterpret_cast<const char *>(&io_timeout),
                static_cast<int>(sizeof(io_timeout))) != 0) {
#else
        const timeval io_timeout{
            static_cast<time_t>(send_idle_deadline.count()),
            0,
        };
        if (::setsockopt(
                native_socket(client.descriptor),
                SOL_SOCKET,
                SO_RCVTIMEO,
                reinterpret_cast<const char *>(&io_timeout),
                static_cast<int>(sizeof(io_timeout))) != 0 ||
            ::setsockopt(
                native_socket(client.descriptor),
                SOL_SOCKET,
                SO_SNDTIMEO,
                reinterpret_cast<const char *>(&io_timeout),
                static_cast<int>(sizeof(io_timeout))) != 0) {
#endif
            std::cerr << "adi: cannot configure client socket: "
                      << "socket error " << socket_error_code() << '\n';
            continue;
        }
        if (!try_acquire_connection(active_connections)) {
            reject_busy(client.descriptor);
            continue;
        }
        try {
            std::thread(
                [client = std::move(client),
                 &model,
                 &batcher,
                 &active_connections]() mutable {
                    try {
                        handle_client(client.descriptor, model, batcher);
                    } catch (...) {
                        std::cerr << "adi: unexpected connection failure\n";
                    }
                    active_connections.fetch_sub(
                        1,
                        std::memory_order_relaxed);
                })
                .detach();
        } catch (const std::exception &error) {
            active_connections.fetch_sub(1, std::memory_order_relaxed);
            std::cerr << "adi: cannot start connection worker: "
                      << error.what() << '\n';
        }
    }
}

} // namespace adi
