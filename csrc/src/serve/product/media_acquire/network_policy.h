#pragma once

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>

namespace sinfer::product::media_acquire::detail {

inline bool private_ipv4(std::uint32_t address) {
    const std::uint32_t a = ntohl(address);
    return (a >> 24U) == 0 || (a >> 24U) == 10 || (a >> 24U) == 127 || (a >> 16U) == 0xa9fe ||
           (a >> 20U) == 0xac1 || (a >> 16U) == 0xc0a8 || (a >> 22U) == 0x0191 ||
           (a & 0xfffe0000U) == 0xc6120000U || (a >> 24U) >= 224; // 198.18.0.0/15
}

inline bool private_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        return private_ipv4(reinterpret_cast<const sockaddr_in*>(address)->sin_addr.s_addr);
    }
    if (address->sa_family != AF_INET6) { return true; }
    const in6_addr& a = reinterpret_cast<const sockaddr_in6*>(address)->sin6_addr;
    if (IN6_IS_ADDR_UNSPECIFIED(&a) || IN6_IS_ADDR_LOOPBACK(&a) || IN6_IS_ADDR_LINKLOCAL(&a) ||
        IN6_IS_ADDR_MULTICAST(&a) || (a.s6_addr[0] & 0xfeU) == 0xfcU) {
        return true;
    }
    if (IN6_IS_ADDR_V4MAPPED(&a)) {
        std::uint32_t v4 = 0;
        std::memcpy(&v4, &a.s6_addr[12], sizeof(v4));
        return private_ipv4(v4);
    }
    return false;
}

} // namespace sinfer::product::media_acquire::detail
