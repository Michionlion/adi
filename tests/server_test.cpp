#include "adi/json.hpp"
#include "server_internal.hpp"

#include <cassert>
#include <chrono>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <unistd.h>
#endif

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

#ifdef _WIN32
    WSADATA winsock{};
    assert(::WSAStartup(MAKEWORD(2, 2), &winsock) == 0);
    const auto listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(listener != INVALID_SOCKET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    assert(::bind(listener, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == 0);
    assert(::listen(listener, 1) == 0);
    int address_length = sizeof(address);
    assert(::getsockname(
               listener,
               reinterpret_cast<sockaddr *>(&address),
               &address_length) == 0);
    const auto client = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(client != INVALID_SOCKET);
    assert(::connect(
               client,
               reinterpret_cast<const sockaddr *>(&address),
               sizeof(address)) == 0);
    const auto server = ::accept(listener, nullptr, nullptr);
    assert(server != INVALID_SOCKET);
    ::closesocket(listener);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    assert(!adi::server_detail::connection_cancelled(
        static_cast<adi::server_detail::SocketHandle>(server), deadline));

    const std::string response = "ok";
    assert(::send(server, response.data(), static_cast<int>(response.size()), 0) ==
           static_cast<int>(response.size()));
    char received[2];
    assert(::recv(client, received, sizeof(received), 0) ==
           static_cast<int>(sizeof(received)));
    assert(std::string(received, sizeof(received)) == response);

    ::closesocket(server);
    ::closesocket(client);
    assert(adi::server_detail::connection_cancelled(
        static_cast<adi::server_detail::SocketHandle>(server),
        std::chrono::steady_clock::now() + std::chrono::seconds(1)));
    ::WSACleanup();
#else
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
#endif
}
