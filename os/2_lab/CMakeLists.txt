cmake_minimum_required(VERSION 3.31)

project(CrossProc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(crossproc STATIC process.cpp)

add_executable(test_code test_code.cpp)

target_link_libraries(test_code PRIVATE crossproc)

set_target_properties(crossproc test_code PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fdiagnostics-color=always)
endif()

message(STATUS "--------------------------------------------")
message(STATUS " Project: CrossProc (C++17)")
message(STATUS " Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " Target OS: ${CMAKE_SYSTEM_NAME}")
message(STATUS "--------------------------------------------")
