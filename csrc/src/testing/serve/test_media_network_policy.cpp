#include "product/media_acquire/network_policy.h"

#include <iostream>
#include <string>
#include <utility>

int main() {
    int failures = 0;
    for (const auto& [ip, blocked] : {
             std::pair{"198.17.255.255", false}, {"198.18.0.0", true}, {"198.18.0.7", true},
             {"198.19.255.255", true}, {"198.20.0.0", false}, {"198.126.0.0", false},
             {"198.127.255.255", false}, {"10.0.0.1", true}, {"127.0.0.1", true},
             {"169.254.0.1", true}, {"172.16.0.1", true}, {"172.31.255.255", true},
             {"192.168.0.1", true}, {"100.64.0.1", true}, {"100.127.255.255", true},
             {"224.0.0.1", true}, {"1.1.1.1", false}, {"8.8.8.8", false}}) {
        for (const bool mapped : {false, true}) {
            sockaddr_storage address{};
            const std::string text = mapped ? "::ffff:" + std::string(ip) : ip;
            if (mapped) {
                auto& v6 = reinterpret_cast<sockaddr_in6&>(address);
                v6.sin6_family = AF_INET6;
                if (inet_pton(AF_INET6, text.c_str(), &v6.sin6_addr) != 1) { return 1; }
            } else {
                auto& v4 = reinterpret_cast<sockaddr_in&>(address);
                v4.sin_family = AF_INET;
                if (inet_pton(AF_INET, text.c_str(), &v4.sin_addr) != 1) { return 1; }
            }
            if (sinfer::product::media_acquire::detail::private_address(
                    reinterpret_cast<const sockaddr*>(&address)) != blocked) {
                ++failures;
                std::cerr << "incorrect media network policy for " << text << '\n';
            }
        }
    }
    return failures != 0;
}
