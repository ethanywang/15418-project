#!/bin/bash
rm -r build
mkdir build && cd build
cmake3 .. && make -j