#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
H=$(node -p 'require("path").resolve(require("path").dirname(process.execPath),"../include/node")')
[ -f "$H/node_api.h" ] || H=/usr/include/node
[ -f "$H/node_api.h" ] || exit 1
mkdir -p build
g++ -O3 -march=native -flto -std=c++20 -pthread -fPIC -shared -Wno-unused-function -Wno-ignored-attributes -I"$H" qne/gf128-v13/push.cc -o build/qne_v13.node
g++ -O3 -march=native -flto -std=c++20 -pthread -fPIC -shared -Wno-unused-function -Wno-ignored-attributes -I"$H" qne-datacenter/gf128-v9/boost.cc -o build/qne_datacenter_v9.node
echo BUILD=PASS
