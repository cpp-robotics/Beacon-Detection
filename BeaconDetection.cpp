/**
 * @author Jonathan Kwong
 * @date 2024-10-05
 * @file BeaconDetection.cpp
 * Description:
 *      A beacon detection program that detects color in a centralized bounding box,
 *      returning a message of whether the target color was detected.
 *      The target color, minimum threshold for a positive detection and
 *      bounding box dimensions can be changed by modifying constant variables.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

// Target Color Properties: specify lower and upper range for the color you want to detect.
//                          The name is for the output message. Color currently set to yellow.
const Scalar LOWER_COLOR(20, 100, 100);  // Lower HSV boundary
const Scalar UPPER_COLOR(30, 255, 255);  // Upper HSV boundary
const string COLOR_NAME = "yellow";

// Central bounding box size (adjust as needed)
const int BOUNDING_BOX_SIZE = 100;  // Size of the box in the center
const int YELLOW_THRESHOLD = 500;  // Number of pixels needed to confirm detection

// Function to check for color in the central bounding box
bool checkColor(const Mat& frame, const Scalar& lowerColor, const Scalar& upperColor,
                int boxSize = 100, int clusterThreshold = 500, Mat* maskOut = nullptr, Mat* dilatedMaskOut = nullptr)
{
    // BGR to HSV Conversion
    Mat hsvFrame;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

    // Creates a mask where the target color is white and the rest are black
    Mat colorMask;
    inRange(hsvFrame, lowerColor, upperColor, colorMask);

    // Dilation to reduce noise (Uses nearest interpolation)
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilatedMask;
    dilate(colorMask, dilatedMask, kernel, Point(-1, -1), 1);

    // Allows visualization of masks when you turn on debugging.
    if (maskOut != nullptr) {
        *maskOut = colorMask;
    }
    if (dilatedMaskOut != nullptr) {
        *dilatedMaskOut = dilatedMask;
    }

    // Centralized Bounding Box
    int centerX = (frame.cols - boxSize) / 2;
    int centerY = (frame.rows - boxSize) / 2;
    Rect centralBox(centerX, centerY, boxSize, boxSize);

    // Draw the bounding box on the frame (for visualization)
    rectangle(frame, centralBox, Scalar(0, 255, 0), 2);

    // Crop the mask using the bounding box
    Mat roi = dilatedMask(centralBox);

    // Count white pixels in the mask (representing the target color)
    int nonZeroCount = countNonZero(roi);

    // If the number of color pixels exceeds the clusterThreshold, return true
    return nonZeroCount > clusterThreshold;
}

int main() {
    // Open the default webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam." << endl;
        return -1;
    }

    // Capture frame-by-frame
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Frame is empty." << endl;
            break;
        }

        // Resizes the frame (320x240) to improve efficiency (using INTER_NEAREST interpolation)
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(320, 240), 0, 0, INTER_NEAREST);

        // Variables for debugging
        Mat mask, dilatedMask;

        // Detects the color in the center of the resized frame
        bool colorDetected = checkColor(resizedFrame, LOWER_COLOR, UPPER_COLOR, BOUNDING_BOX_SIZE, YELLOW_THRESHOLD, &mask, &dilatedMask);

        if (colorDetected) {
            cout << COLOR_NAME << " detected!" << endl;  // Output the dynamic color name
        }
        else {
            cout << "No " << COLOR_NAME << " detected." << endl;
        }

        // Display windows for debugging (comment out these sections for final use)
         imshow("Resized Frame", resizedFrame);           // Display resized frame
        // imshow("Color Mask", mask);                      // Display mask after applying color filter
         imshow("Dilated Mask", dilatedMask);             // Display dilated mask after removing noise

        // Delay to control the scanning speed (milliseconds)
        this_thread::sleep_for(chrono::milliseconds(100));

        // Exit if 'q' is pressed
        if (waitKey(30) == 'q') {
            break;
        }
    }

    // Release the webcam and close all OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}
