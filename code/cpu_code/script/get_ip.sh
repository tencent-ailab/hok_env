#!/usr/bin/env bash
function query_eth_name() {
    ip route get 10.0.0.0 | head -n 1 | awk '{ \
        for (i = 1; i < NF; ++i) { \
            if ($i == "dev") { \
                print $(i + 1); \
                break; \
            } \
        } \
    }'
}
function query_client_ip() {
    ip -o -4 addr show dev $(query_eth_name) | head -n 1 | awk '{ \
        for (i = 1; i < NF; ++i) { \
            if ($i == "inet") { \
                print gensub(/([^\/]*)\/.*/, "\\1", "g", $(i + 1)); \
                break; \
            } \
        } \
    }'
}

export local_host_ip=$(query_client_ip)
echo ${local_host_ip}
