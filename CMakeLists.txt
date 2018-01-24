cmake_minimum_required(VERSION 3.8)
project(cv_project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -fopenmp")
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp MeanshiftTracker.cpp MeanshiftTracker.h LucasKanadeTracker.cpp LucasKanadeTracker.h Tracker.h)
add_executable(cv_project ${SOURCE_FILES})
target_link_libraries(cv_project ${OpenCV_LIBS})