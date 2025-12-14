# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(WeaponDetectionSystem LANGUAGES CXX)

# Specify the C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define an executable target, listing its source files
set(WDS_SOURCES 
    src/main.cpp 
)

# Define an executable target
add_executable(wds_program ${WDS_SOURCES})

find_package(OpenCV REQUIRED)


# 1. Find the Boost package and components
# Note: FindBoost is deprecated (as noted by the warning), but still works
# for now.
find_package(Boost 1.89.0 REQUIRED  )

# 2. Add include directories for headers
# For OpenCV: include_directories is okay, but this is better:
target_include_directories(wds_program PRIVATE ${OpenCV_INCLUDE_DIRS})

# For Boost:
target_include_directories(wds_program PRIVATE ${Boost_INCLUDE_DIRS})


# 3. Link the Boost libraries using the modern target syntax
# This is the crucial step that links the required library.
target_link_libraries(wds_program PRIVATE 
    ${OpenCV_LIBS}         
    ${Boost_LIBRARIES}      
)