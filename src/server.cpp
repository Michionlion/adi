#include "adi/server.hpp"

#include "adi/generation.hpp"
#include "adi/json.hpp"
#include "chat.hpp"
#include "server_internal.hpp"
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace adi {
namespace {

constexpr std::size_t maximum_headers = 64 * 1024;
constexpr std::size_t maximum_body = 1024 * 1024;
constexpr auto header_deadline = std::chrono::seconds(15);
constexpr auto body_deadline = std::chrono::seconds(30);
constexpr auto send_idle_deadline = std::chrono::seconds(15);
constexpr auto request_deadline = std::chrono::minutes(30);
std::atomic<std::uint64_t> response_counter = 0;
using TimePoint = std::chrono::steady_clock::time_point;

struct Socket {
    int descriptor = -1;
    ~Socket() {
        if (descriptor >= 0) {
            ::close(descriptor);
        }
    }
};

class SocketFailure : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct Connection {
    int socket;
    TimePoint deadline;
};

using server_detail::ResponseIdentity;
using server_detail::connection_cancelled;
using server_detail::response_json;

[[noreturn]] void socket_error(std::string_view action) {
    throw SocketFailure(
        std::string(action) + ": " + std::strerror(errno));
}

void wait_for_socket(int socket, short events, TimePoint deadline) {
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw SocketFailure("socket deadline exceeded");
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto timeout = static_cast<int>(std::clamp<std::int64_t>(
            remaining.count(), 1, std::numeric_limits<int>::max()));
        pollfd descriptor{socket, events, 0};
        const auto result = ::poll(&descriptor, 1, timeout);
        if (result > 0) {
            if ((descriptor.revents & events) != 0) {
                return;
            }
            if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                throw SocketFailure("socket disconnected");
            }
        } else if (result == 0) {
            throw SocketFailure("socket deadline exceeded");
        } else if (errno != EINTR) {
            socket_error("poll");
        }
    }
}

void send_all(Connection &connection, std::string_view data) {
    auto idle = std::min(
        connection.deadline,
        std::chrono::steady_clock::now() + send_idle_deadline);
    while (!data.empty()) {
        wait_for_socket(connection.socket, POLLOUT, idle);
        const auto sent = ::send(
            connection.socket,
            data.data(),
            data.size(),
            MSG_NOSIGNAL | MSG_DONTWAIT);
        if (sent < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
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
        [](unsigned char character) { return std::tolower(character); });
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
    int socket,
    std::span<char> buffer,
    TimePoint deadline) {
    wait_for_socket(socket, POLLIN, deadline);
    for (;;) {
        const auto received =
            ::recv(socket, buffer.data(), buffer.size(), MSG_DONTWAIT);
        if (received > 0) {
            return static_cast<std::size_t>(received);
        }
        if (received == 0) {
            throw SocketFailure("client disconnected while sending request");
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            wait_for_socket(socket, POLLIN, deadline);
            continue;
        }
        socket_error("receive");
    }
}

std::string receive_request(int socket, TimePoint overall_deadline) {
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

std::string formatted_input(const Json &root) {
    const auto *input = root.find("input");
    if (input == nullptr) {
        throw std::runtime_error("'input' is required");
    }
    if (const auto *text = input->string()) {
        return qwen_user_prompt(*text);
    }
    const auto *messages = input->array();
    if (messages == nullptr || messages->empty()) {
        throw std::runtime_error("'input' must be a string or non-empty message array");
    }
    std::vector<ChatMessage> rendered_messages;
    rendered_messages.reserve(messages->size());
    for (const auto &message : *messages) {
        const auto *role = message.find("role");
        const auto *content = message.find("content");
        if (role == nullptr || role->string() == nullptr || content == nullptr) {
            throw std::runtime_error("each input message needs role and content");
        }
        if (*role->string() != "user" && *role->string() != "assistant" &&
            *role->string() != "system") {
            throw std::runtime_error("unsupported input message role");
        }
        rendered_messages.push_back(
            {*role->string(), content_text(*content)});
    }
    return qwen35_chat_prompt(rendered_messages);
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

std::string response_json(
    std::string_view model_name,
    const GenerationResult &result,
    const ResponseIdentity &identity) {
    const bool incomplete = result.finish_reason == FinishReason::length;
    const std::string_view response_status =
        incomplete ? "incomplete" : "completed";
    const std::string_view item_status =
        incomplete ? "incomplete" : "completed";
    const std::string_view incomplete_details =
        incomplete ? "{\"reason\":\"max_output_tokens\"}" : "null";
    return "{\"id\":" + json_string(identity.id) +
           ",\"object\":\"response\",\"created_at\":" +
           std::to_string(identity.created_at) +
           ",\"status\":" + json_string(response_status) +
           ",\"error\":null,\"incomplete_details\":" +
           std::string(incomplete_details) +
           ",\"model\":" + json_string(model_name) +
           ",\"output\":[{\"id\":" + json_string(identity.message_id) +
           ",\"type\":\"message\",\"status\":" + json_string(item_status) +
           ",\"role\":\"assistant\""
           ",\"content\":[{\"type\":\"output_text\",\"text\":" +
           json_string(result.text) +
           ",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":" +
           std::to_string(result.input_tokens) +
           ",\"output_tokens\":" + std::to_string(result.output_tokens) +
           ",\"total_tokens\":" +
           std::to_string(result.input_tokens + result.output_tokens) + "}}";
}

bool connection_cancelled(
    int socket,
    std::chrono::steady_clock::time_point deadline) noexcept {
    if (std::chrono::steady_clock::now() >= deadline) {
        return true;
    }
    pollfd descriptor{
        socket,
        static_cast<short>(POLLIN | POLLERR | POLLHUP),
        0,
    };
    const auto ready = ::poll(&descriptor, 1, 0);
    if (ready < 0) {
        return errno != EINTR;
    }
    if (ready == 0) {
        return false;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return true;
    }
    if ((descriptor.revents & POLLIN) != 0) {
        char byte = 0;
        const auto received =
            ::recv(socket, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        // Read EOF includes a valid peer shutdown(SHUT_WR). A failed response
        // send will detect whether the peer also stopped receiving.
        return received < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
               errno != EINTR;
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
    options.max_output_tokens =
        integer_option(request, "max_output_tokens", options.max_output_tokens);
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

void validate_features(const Json &request) {
    if (const auto *tools = request.find("tools");
        tools != nullptr && (!tools->is_null() &&
                             (tools->array() == nullptr || !tools->array()->empty()))) {
        throw std::runtime_error("tools are not supported");
    }
}

void send_event(
    Connection &connection,
    std::string_view event,
    std::string data) {
    send_all(connection, "event: " + std::string(event) + "\n");
    send_all(connection, "data: " + data + "\n\n");
}

void stream_responses(
    Connection &connection,
    const Json &request,
    const MachModel &model,
    Tokenizer &tokenizer) {
    const auto headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    send_all(connection, headers);
    try {
        const auto identity = make_identity();
        const std::string model_name(request_model(request, model));
        std::uint32_t sequence = 0;
        const auto response_stub =
            "{\"id\":" + json_string(identity.id) +
            ",\"object\":\"response\",\"created_at\":" +
            std::to_string(identity.created_at) +
            ",\"status\":\"in_progress\",\"model\":" +
            json_string(model_name) + ",\"output\":[]}";
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
        send_event(
            connection,
            "response.output_item.added",
            "{\"type\":\"response.output_item.added\",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"output_index\":0,\"item\":{\"id\":" +
                json_string(identity.message_id) +
                ",\"type\":\"message\",\"status\":\"in_progress\","
                "\"role\":\"assistant\",\"content\":[]}}");
        send_event(
            connection,
            "response.content_part.added",
            "{\"type\":\"response.content_part.added\",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"item_id\":" + json_string(identity.message_id) +
                ",\"output_index\":0,\"content_index\":0,\"part\":"
                "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}");

        std::string pending;
        const auto emit_delta = [&](std::string_view delta) {
            if (delta.empty()) {
                return;
            }
            send_event(
                connection,
                "response.output_text.delta",
                "{\"type\":\"response.output_text.delta\",\"sequence_number\":" +
                    std::to_string(sequence++) +
                    ",\"item_id\":" + json_string(identity.message_id) +
                    ",\"output_index\":0,\"content_index\":0,\"delta\":" +
                    json_string(delta) + "}");
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
        const auto result = generate_from_prompt(
            model,
            tokenizer,
            formatted_input(request),
            request_options(request),
            [&](std::string_view piece) {
                pending.append(piece);
                drain_pending(false);
            },
            cancellation_check);
        drain_pending(true);
        send_event(
            connection,
            "response.output_text.done",
            "{\"type\":\"response.output_text.done\",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"item_id\":" + json_string(identity.message_id) +
                ",\"output_index\":0,\"content_index\":0,\"text\":" +
                json_string(result.text) + "}");
        send_event(
            connection,
            "response.content_part.done",
            "{\"type\":\"response.content_part.done\",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"item_id\":" + json_string(identity.message_id) +
                ",\"output_index\":0,\"content_index\":0,\"part\":"
                "{\"type\":\"output_text\",\"text\":" +
                json_string(result.text) + ",\"annotations\":[]}}");
        send_event(
            connection,
            "response.output_item.done",
            "{\"type\":\"response.output_item.done\",\"sequence_number\":" +
                std::to_string(sequence++) +
                ",\"output_index\":0,\"item\":{\"id\":" +
                json_string(identity.message_id) +
                ",\"type\":\"message\",\"status\":" +
                json_string(
                    result.finish_reason == FinishReason::length
                        ? "incomplete"
                        : "completed") +
                ","
                "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\","
                "\"text\":" + json_string(result.text) +
                ",\"annotations\":[]}]}}");
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
                response_json(model_name, result, identity) + "}");
    } catch (const SocketFailure &) {
        return;
    } catch (const std::exception &error) {
        try {
            send_event(
                connection,
                "error",
                "{\"type\":\"error\",\"error\":{\"message\":" +
                    json_string(error.what()) +
                    ",\"type\":\"invalid_request_error\"}}");
        } catch (const SocketFailure &) {
        }
    }
}

std::string handle_responses(
    const Json &request,
    const MachModel &model,
    Tokenizer &tokenizer,
    Connection &connection) {
    if (request.object() == nullptr) {
        throw std::runtime_error("request body must be a JSON object");
    }
    if (const auto *stream = request.find("stream");
        stream != nullptr && stream->boolean() == nullptr) {
        throw std::runtime_error("'stream' must be a boolean");
    }
    validate_features(request);
    const auto model_name = request_model(request, model);
    const auto prompt = formatted_input(request);
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
    const auto result = generate_from_prompt(
        model,
        tokenizer,
        prompt,
        request_options(request),
        {},
        cancellation_check);
    const auto identity = make_identity();
    return response_json(model_name, result, identity);
}

void handle_client(
    int socket,
    const MachModel &model,
    Tokenizer &tokenizer) noexcept {
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
        validate_features(parsed);
        (void)request_model(parsed, model);
        if (const auto *stream = parsed.find("stream");
            stream != nullptr && stream->boolean() != nullptr && *stream->boolean()) {
            stream_responses(connection, parsed, model, tokenizer);
        } else {
            send_response(
                connection,
                200,
                "OK",
                handle_responses(parsed, model, tokenizer, connection));
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
    const MachModel model(options.model);
    Tokenizer tokenizer(model);
    Socket listener{::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0)};
    if (listener.descriptor < 0) {
        socket_error("socket");
    }
    int reuse = 1;
    if (::setsockopt(
            listener.descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        socket_error("setsockopt");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    if (::inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("server: host must be an IPv4 address");
    }
    if (::bind(
            listener.descriptor,
            reinterpret_cast<const sockaddr *>(&address),
            sizeof(address)) != 0) {
        socket_error("bind");
    }
    if (::listen(listener.descriptor, 16) != 0) {
        socket_error("listen");
    }
    std::cout << "adi: listening on http://" << options.host << ':' << options.port
              << "/v1/responses\n";
    std::cout.flush();
    for (;;) {
        Socket client{::accept4(listener.descriptor, nullptr, nullptr, SOCK_CLOEXEC)};
        if (client.descriptor < 0) {
            if (errno == EINTR) {
                continue;
            }
            socket_error("accept");
        }
        const timeval io_timeout{
            static_cast<time_t>(send_idle_deadline.count()),
            0,
        };
        if (::setsockopt(
                client.descriptor,
                SOL_SOCKET,
                SO_RCVTIMEO,
                &io_timeout,
                sizeof(io_timeout)) != 0 ||
            ::setsockopt(
                client.descriptor,
                SOL_SOCKET,
                SO_SNDTIMEO,
                &io_timeout,
                sizeof(io_timeout)) != 0) {
            std::cerr << "adi: cannot configure client socket: "
                      << std::strerror(errno) << '\n';
            continue;
        }
        try {
            handle_client(client.descriptor, model, tokenizer);
        } catch (...) {
            std::cerr << "adi: unexpected connection failure\n";
        }
    }
}

} // namespace adi
