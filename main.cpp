//
// Created by Patrick Werner (boonto) on 10.10.17.
//

#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <thread>
#include "MeanshiftTracker.h"
#include "LucasKanadeTracker.h"

// Handle input arguments to avoid rebuilding for parameter changes
// TODO: Pretty bad
int parseArguments(int argc, char *argv[], std::string &videoPath, std::string &groundTruthFileName,
                   std::string &errorFileName, bool &bUseGroundTruth, bool &bWriteErrorToFile, int &currentTracker) {
    if (2 == argc) {
        auto argumentsFile = std::ifstream(argv[1]);
        if (argumentsFile.fail()) {
            std::cerr << "Error opening arguments file: " << std::strerror(errno) << "\n";
            std::cerr << "Press a button to quit.\n";
            std::cin.get();
            return 1;
        }
        auto line = std::string();
        while (std::getline(argumentsFile, line)) {
            if (line.front() != '#') {
                auto argName = line.substr(0, line.find('='));
                auto argValue = line.substr(line.find('=') + 1);
                if (argName == "videoPath") {
                    videoPath = argValue;
                } else if (argName == "groundTruthFileName") {
                    groundTruthFileName = argValue;
                } else if (argName == "errorFileName") {
                    errorFileName = argValue;
                } else if (argName == "bUseGroundTruth") {
                    bUseGroundTruth = (argValue == "true");
                } else if (argName == "bWriteErrorToFile") {
                    bWriteErrorToFile = (argValue == "true");
                } else if (argName == "currentTracker") {
                    currentTracker = std::stoi(argValue);
                }
            }
        }
    } else if (7 == argc) {
        videoPath = argv[1];
        groundTruthFileName = argv[2];
        errorFileName = argv[3];
        bUseGroundTruth = (std::string(argv[4]) == "true");
        bWriteErrorToFile = (std::string(argv[5]) == "true");
        currentTracker = std::stoi(argv[6]);
    } else {
        std::cerr << "Please specify a arguments file!\n";
        std::cerr << "Press a button to quit.\n";
        std::cin.get();
        return 1;
    }

    return 0;
}

// Converts a line in a text file to a rect
cv::Rect2f lineToRect(std::ifstream &stream, int offset) {
    auto line = std::string();
    std::getline(stream, line);
    auto lineStream = std::stringstream(line);
    auto item = std::string();
    auto points = std::vector<int>();
    auto prev = std::size_t();
    decltype(prev) pos;
    while ((pos = line.find_first_of(",\t", prev)) != std::string::npos) {
        if (pos > prev) {
            points.push_back(std::stoi(line.substr(prev, pos - prev)));
        }
        prev = pos + 1;
    }
    if (prev < line.length()) {
        points.push_back(std::stoi(line.substr(prev, std::string::npos)));
    }
    if (!points.empty()) {
        return cv::Rect2f(points[0] - offset, points[1] - offset, points[2] + offset * 2, points[3] + offset * 2);
    } else {
        return cv::Rect2f();
    }
}

int main(int argc, char *argv[]) {
    // Various arguments
    auto videoPath = std::string();
    auto groundTruthFileName = std::string();
    auto errorFileName = std::string();

    auto bUseGroundTruth = false;
    auto bWriteErrorToFile = false;

    // Starting tracker 0 = LK, 1 = MS
    auto currentTracker = 0;

    if (parseArguments(argc, argv, videoPath, groundTruthFileName, errorFileName,
                       bUseGroundTruth, bWriteErrorToFile, currentTracker) != 0) {
        return EXIT_FAILURE;
    }

    // Timing
    auto t0 = std::chrono::high_resolution_clock::now();

    // Initialize video
    auto vidCap = cv::VideoCapture();
    if (videoPath.empty()) {
        // Use camera
        vidCap.open(0);
        // Discard the first few frames to let the camera adjust to surroundings
        for (int i = 0; i < 15; ++i) {
            vidCap.grab();
        }
    } else {
        // Use video
        vidCap.open(videoPath);
    }

    if (!vidCap.isOpened()) {
        std::cerr << "Failed to open video\n";
        return EXIT_FAILURE;
    }

    // Load ground truth file
    auto groundTruthFile = std::ifstream();
    auto lastPos = videoPath.find_last_of('/');
    auto secondToLastPos = videoPath.substr(0, lastPos).find_last_of('/');
    groundTruthFile.open(videoPath.substr(0, secondToLastPos + 1) + groundTruthFileName);
    if (groundTruthFile.fail()) {
        std::cerr << "Error opening ground truth file: " << std::strerror(errno) << "\n"
                  << "Ignoring ground truth data\n";
        bUseGroundTruth = false;
    }

    auto resolution = cv::Point(static_cast<int>(vidCap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                static_cast<int>(vidCap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    std::cout << "Video frame size is " << resolution << "\n";

    std::string windowName = "Tracking";
    cv::namedWindow(windowName);

    // Initialize overarching variables
    auto frame = cv::Mat();
    auto display = cv::Mat();
    auto roi = cv::Rect2f(cv::Point2f(), resolution);
    auto bRoiInitialized = false;

    auto trackers = std::vector<std::unique_ptr<Tracker>>();
    trackers.push_back(std::unique_ptr<Tracker>(new LucasKanadeTracker(LucasKanadeTracker::Parameters())));
    trackers.push_back(std::unique_ptr<Tracker>(new MeanshiftTracker(MeanshiftTracker::Parameters())));

    // Load error writing file
    errorFileName += trackers[currentTracker]->classname();
    auto errorFile = std::ofstream();
    if (bUseGroundTruth && bWriteErrorToFile) {
        errorFile.open(videoPath.substr(0, secondToLastPos + 1) + errorFileName);
        if (errorFile.fail()) {
            std::cerr << "Error opening error file" << std::strerror(errno) << "\n";
            bWriteErrorToFile = false;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms";

    while (true) {
        // Get next frame
        vidCap >> frame;

        // Restart video when it is over
        if (frame.empty()) {
            vidCap.set(cv::CAP_PROP_POS_FRAMES, 0);
            vidCap >> frame;
            trackers[currentTracker]->reset();
            bRoiInitialized = false;
            // Reset ground truth file
            if (bUseGroundTruth) {
                groundTruthFile.clear();
                groundTruthFile.seekg(0, std::ios::beg);

                // Stop recording errors
                if (bWriteErrorToFile) {
                    bWriteErrorToFile = false;
                }
            }
        }

        // Mat for displaying in the window
        display = frame.clone();

        // Initialize roi
        if (!bRoiInitialized) {
            // Use ground truth roi if available
            if (bUseGroundTruth) {
                roi = lineToRect(groundTruthFile, 0);
                std::cout << "\nStarting roi " << roi << "\n";
                // To avoid 1 line difference
                groundTruthFile.clear();
                groundTruthFile.seekg(0, std::ios::beg);
            } else {
                roi = cv::selectROI(frame);
                std::cout << "\nSelected roi " << roi << "\n";
                if (roi.width == 0 || roi.height == 0) {
                    std::cerr << "No ROI selected\n Ignoring roi\n";
                }
                trackers[currentTracker]->reset();
            }
            bRoiInitialized = true;
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // Use the current tracker to track the region of interest
        trackers[currentTracker]->track(frame, roi);

        // Display tracking points if its the Lucas Kanade tracker
        if (dynamic_cast<LucasKanadeTracker *>(trackers[currentTracker].get())) {
            dynamic_cast<LucasKanadeTracker *>(trackers[currentTracker].get())->display(display);
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "\r" << "Tracking: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " ms";
        std::cout.flush();

        // Show roi
        cv::rectangle(display, roi, cv::Scalar(0, 255, 255));

        // Compare roi with ground truth roi
        if(bUseGroundTruth) {
            auto groundTruthRoi = lineToRect(groundTruthFile, 0);

            if (bWriteErrorToFile) {
                errorFile << trackers[currentTracker]->evaluate(roi, groundTruthRoi) << "\n";
            } /*else {
                std::cout << trackers[currentTracker]->classname() << " "
                          << trackers[currentTracker]->evaluate(roi, groundTruthRoi) << "\n";
            }*/

            // Show ground truth roi
            cv::rectangle(display, groundTruthRoi, cv::Scalar(0, 0, 255));
        }

        // Display mat in window
        cv::imshow(windowName, display);

        // ~ 30 fps
        int keyPressed = cv::waitKey(33);
        if (keyPressed != -1) {
            // Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states.
            keyPressed &= 0xFFFF;
            std::cout << "\nKey pressed: " << keyPressed << "\n";
            // Quit the loop when the Esc key is pressed.
            if (keyPressed == 27)
                break;
            // Space
            if (keyPressed == 32) {
                currentTracker = (currentTracker + 1) % static_cast<int>(trackers.size());
                trackers[currentTracker]->reset();
            }
            // F
            if (keyPressed == 102)
                bRoiInitialized = false;
            // G
//            if (keyPressed == 103)
        }
    }

    groundTruthFile.close();
    errorFile.close();

    return EXIT_SUCCESS;
}
