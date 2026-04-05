# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(WeaponDetectionSystem LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

option(ENABLE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined)
    message(STATUS "Sanitizers enabled: Address, Undefined Behavior")
endif()

# Specify the C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define an executable target, listing its source files
set(WDS_SOURCES 
    src/main.cpp 
)


# Define an executable target
add_executable(wds_program ${WDS_SOURCES})

find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)


# 1. Find the Boost package and components
# Note: FindBoost is deprecated (as noted by the warning), but still works
# for now.
find_package(Boost 1.89.0 REQUIRED  )

target_link_libraries(wds_program PRIVATE OpenSSL::SSL OpenSSL::Crypto 
  CURL::libcurl Boost::headers Threads::Threads
    ${OpenCV_LIBS}         )

