#include "adi/json.hpp"
#include "server_internal.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
using TestSocket = SOCKET;

int send_some(TestSocket socket, const char *data, std::size_t size) {
    return ::send(socket, data, static_cast<int>(size), 0);
}

int receive_some(TestSocket socket, char *data, std::size_t size) {
    return ::recv(socket, data, static_cast<int>(size), 0);
}

void close_test_socket(TestSocket socket) {
    assert(::closesocket(socket) == 0);
}
#else
using TestSocket = int;

ssize_t send_some(TestSocket socket, const char *data, std::size_t size) {
    return ::send(socket, data, size, 0);
}

ssize_t receive_some(TestSocket socket, char *data, std::size_t size) {
    return ::recv(socket, data, size, 0);
}

void close_test_socket(TestSocket socket) {
    assert(::close(socket) == 0);
}
#endif

void send_all(TestSocket socket, std::string_view data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto sent = send_some(socket, data.data() + offset, data.size() - offset);
        assert(sent > 0);
        offset += static_cast<std::size_t>(sent);
    }
}

void receive_all(TestSocket socket, std::span<char> data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto received =
            receive_some(socket, data.data() + offset, data.size() - offset);
        assert(received > 0);
        offset += static_cast<std::size_t>(received);
    }
}

} // namespace

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

    TestSocket client;
    TestSocket server;
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
    client = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(client != INVALID_SOCKET);
    assert(::connect(
               client,
               reinterpret_cast<const sockaddr *>(&address),
               sizeof(address)) == 0);
    server = ::accept(listener, nullptr, nullptr);
    assert(server != INVALID_SOCKET);
    close_test_socket(listener);
#else
    int sockets[2];
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(::shutdown(sockets[1], SHUT_WR) == 0);
    server = sockets[0];
    client = sockets[1];
#endif

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    assert(!adi::server_detail::connection_cancelled(
        static_cast<adi::server_detail::SocketHandle>(server), deadline));

    const std::string response = "ok";
    send_all(server, response);
    char received[2];
    receive_all(client, received);
    assert(std::string(received, sizeof(received)) == response);

    const auto closed_socket =
        static_cast<adi::server_detail::SocketHandle>(server);
    close_test_socket(server);
    close_test_socket(client);
    assert(adi::server_detail::connection_cancelled(
        closed_socket,
        std::chrono::steady_clock::now() + std::chrono::seconds(1)));
#ifdef _WIN32
    assert(::WSACleanup() == 0);
#endif
}
