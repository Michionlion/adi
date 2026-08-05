#include "adi/server.hpp"

#include "adi/generation.hpp"
#include "adi/json.hpp"

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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace adi {
namespace {

constexpr std::size_t maximum_headers = 64 * 1024;
constexpr std::size_t maximum_body = 1024 * 1024;
std::atomic<std::uint64_t> response_counter = 0;

struct Socket {
    int descriptor = -1;
    ~Socket() {
        if (descriptor >= 0) {
            ::close(descriptor);
        }
    }
};

[[noreturn]] void socket_error(std::string_view action) {
    throw std::runtime_error(
        std::string(action) + ": " + std::strerror(errno));
}

void send_all(int socket, std::string_view data) {
    while (!data.empty()) {
        const auto sent = ::send(socket, data.data(), data.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            socket_error("send");
        }
        data.remove_prefix(static_cast<std::size_t>(sent));
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

std::string receive_request(int socket) {
    std::string request;
    char buffer[8192];
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        const auto received = ::recv(socket, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            throw std::runtime_error("client disconnected while sending headers");
        }
        request.append(buffer, static_cast<std::size_t>(received));
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
    while (request.size() < total) {
        const auto received = ::recv(socket, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            throw std::runtime_error("client disconnected while sending body");
        }
        request.append(buffer, static_cast<std::size_t>(received));
    }
    request.resize(total);
    return request;
}

std::string error_body(std::string_view message) {
    return "{\"error\":{\"message\":" + json_string(message) +
           ",\"type\":\"invalid_request_error\"}}";
}

void send_response(
    int socket,
    int status,
    std::string_view reason,
    std::string body) {
    const auto headers =
        "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    send_all(socket, headers);
    send_all(socket, body);
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
    std::string prompt;
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
        prompt += "<|im_start|>" + *role->string() + "\n" +
                  content_text(*content) + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n<think>\n";
    return prompt;
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
    return static_cast<float>(*number);
}

std::string response_json(
    std::string_view model_name,
    const GenerationResult &result) {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    const auto sequence = response_counter.fetch_add(1);
    const auto id = "resp_" + std::to_string(now) + "_" + std::to_string(sequence);
    const auto message_id = "msg_" + std::to_string(now) + "_" + std::to_string(sequence);
    return "{\"id\":" + json_string(id) +
           ",\"object\":\"response\",\"created_at\":" + std::to_string(now) +
           ",\"status\":\"completed\",\"error\":null,\"incomplete_details\":null"
           ",\"model\":" + json_string(model_name) +
           ",\"output\":[{\"id\":" + json_string(message_id) +
           ",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\""
           ",\"content\":[{\"type\":\"output_text\",\"text\":" +
           json_string(result.text) +
           ",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":" +
           std::to_string(result.input_tokens) +
           ",\"output_tokens\":" + std::to_string(result.output_tokens) +
           ",\"total_tokens\":" +
           std::to_string(result.input_tokens + result.output_tokens) + "}}";
}

std::string handle_responses(
    const Json &request,
    const MachModel &model,
    Tokenizer &tokenizer) {
    if (request.object() == nullptr) {
        throw std::runtime_error("request body must be a JSON object");
    }
    if (const auto *stream = request.find("stream");
        stream != nullptr && (stream->boolean() == nullptr || *stream->boolean())) {
        throw std::runtime_error("streaming is not supported yet");
    }
    if (const auto *tools = request.find("tools");
        tools != nullptr && (!tools->is_null() &&
                             (tools->array() == nullptr || !tools->array()->empty()))) {
        throw std::runtime_error("tools are not supported");
    }
    GenerationOptions options;
    options.max_output_tokens =
        integer_option(request, "max_output_tokens", options.max_output_tokens);
    options.temperature = float_option(request, "temperature", options.temperature);
    options.top_p = float_option(request, "top_p", options.top_p);
    options.seed = integer_option(request, "seed", 0);
    const auto prompt = formatted_input(request);
    const auto result =
        generate_from_prompt(model, tokenizer, prompt, options);
    std::string_view model_name = "Mach-1-Additive-35B";
    if (const auto *requested = request.find("model");
        requested != nullptr && requested->string() != nullptr) {
        model_name = *requested->string();
    }
    return response_json(model_name, result);
}

void handle_client(int socket, const MachModel &model, Tokenizer &tokenizer) {
    try {
        const auto request = receive_request(socket);
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
            send_response(socket, 404, "Not Found", error_body("route not found"));
            return;
        }
        const auto body_position = request.find("\r\n\r\n") + 4;
        const auto body = std::string_view(request).substr(body_position);
        send_response(
            socket,
            200,
            "OK",
            handle_responses(parse_json(body), model, tokenizer));
    } catch (const std::exception &error) {
        send_response(socket, 400, "Bad Request", error_body(error.what()));
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
        handle_client(client.descriptor, model, tokenizer);
    }
}

} // namespace adi
