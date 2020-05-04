cmake_minimum_required(VERSION 3.8)
project(tracking)

if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GCC detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -fopenmp")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp MeanshiftTracker.cpp MeanshiftTracker.h LucasKanadeTracker.cpp LucasKanadeTracker.h Tracker.h)
add_executable(tracking ${SOURCE_FILES})
target_link_libraries(tracking ${OpenCV_LIBS})