cmake_minimum_required(VERSION 3.16)
project(TemperatureMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED COMPONENTS Widgets Network)

add_executable(TemperatureMonitor
    main.cpp
    mainwindow.cpp
    mainwindow.h
    qtwebclient.cpp
    qtwebclient.h
)

include_directories(/usr/include/qt5/qwt)

target_link_libraries(TemperatureMonitor
    Qt5::Widgets
    Qt5::Network
    qwt-qt5
)
