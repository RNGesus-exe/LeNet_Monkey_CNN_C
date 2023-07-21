cmake_minimum_required(VERSION 3.5)
project(Lenet_monkey_cnn VERSION 1.0)

# Version of C++ used
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(JPEG REQUIRED)

# OpenCV
find_package(OpenCV 4 REQUIRED)
# !OpenCV

# Add your source files
set(SOURCES
    src/main.cpp
    src/cnn.cpp
)

# Add your header files
set(HEADERS
    include/cnn.h
)

# Specify the include directories
include_directories(include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${JPEG_LIBRARIES})
