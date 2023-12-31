#!/bin/bash

set -e

cd cpp

for w in $(seq 6 11);
do
for h in $(seq 6 11);
do
meson setup -Dcpp_args="-DQUORIDOR_WIDTH=$w -DQUORIDOR_HEIGHT=$h" build --buildtype=release --reconfigure
meson test -C build --print-errorlogs game_quoridor
done
done
