#!/bin/bash

RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

SCRIPTS_DIR="$(cd "$(dirname "$0")"; pwd)"
BUILD_DIR="$SCRIPTS_DIR/build"

cd "$SCRIPTS_DIR"

if ! [ -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
  echo -e "${YELLOW}===\t MAKE BUILT DIR \t==="
fi

cd "$BUILD_DIR"

echo -e "${GREEN}===\t CONFIGURE CMAKE \t==="
echo -e "${RESET}Choose build type"
echo "1) Debug"
echo "2) Release"
read -p "Your choice (Default - Release): " choice

if [ -z "$choice" ]; then
  build_type="Release"
elif [ "$choice" == "1" ]; then
  build_type="Debug"
else
  build_type="Release"
fi 

echo -e "Configure in $build_type mode"

cmake -DCMAKE_BUILD_TYPE=$build_type ..

echo -e "${GREEN}+++\t CMAKE IS COMPILED \t+++"


echo -e "${GREEN}===\t BUILDING PROJECT \t==="

cmake --build . --config $build_type

echo -e "${GREEN}+++\t BUILDING COMPLETE \t+++"


echo -e "${GREEN}===\t START UP PROJECT \t==="
if [ -x "./forked" ]; then
  ./forked
else
  echo -e "${RED}NO SUCH FILE: $BUILD_DIR/hello_world"
fi
