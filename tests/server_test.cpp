#include "adi/json.hpp"
#include "server_internal.hpp"

#include <cassert>
#include <chrono>
#include <string>

#include <sys/socket.h>
#include <unistd.h>

int main() {
    const adi::server_detail::ResponseIdentity identity{
        "resp_test",
        "msg_test",
        123,
    };
    const adi::GenerationResult limited{
        "answer",
        4,
        2,
        adi::FinishReason::length,
    };
    const auto limited_json =
        adi::server_detail::response_json("model", limited, identity);
    const auto limited_response = adi::parse_json(limited_json);
    assert(*limited_response.find("status")->string() == "incomplete");
    assert(
        *limited_response.find("incomplete_details")
             ->find("reason")
             ->string() == "max_output_tokens");
    assert(limited_json.find("\"reason\":\"max_tokens\"") ==
           std::string::npos);

    auto completed = limited;
    completed.finish_reason = adi::FinishReason::stop_token;
    const auto completed_response = adi::parse_json(
        adi::server_detail::response_json("model", completed, identity));
    assert(completed_response.find("incomplete_details")->is_null());

    int sockets[2];
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(::shutdown(sockets[1], SHUT_WR) == 0);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    assert(!adi::server_detail::connection_cancelled(sockets[0], deadline));

    const std::string response = "ok";
    assert(::send(sockets[0], response.data(), response.size(), 0) ==
           static_cast<ssize_t>(response.size()));
    char received[2];
    assert(::recv(sockets[1], received, sizeof(received), 0) ==
           static_cast<ssize_t>(sizeof(received)));
    assert(std::string(received, sizeof(received)) == response);

    const int closed_socket = sockets[0];
    ::close(closed_socket);
    ::close(sockets[1]);
    assert(adi::server_detail::connection_cancelled(
        closed_socket,
        std::chrono::steady_clock::now() + std::chrono::seconds(1)));
}
