#!/bin/bash

set -e

cd cpp
rm -rf ./build

for w in $(seq 6 11);
do
for h in $(seq 6 11);
do
meson setup -Dcpp_args="-ffast-math -march=native -DQUORIDOR_WIDTH=$w -DQUORIDOR_HEIGHT=$h" build --buildtype=release
meson compile -C build
meson test -C build --print-errorlogs
done
done
