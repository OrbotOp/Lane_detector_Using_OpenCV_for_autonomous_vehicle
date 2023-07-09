#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "/home/roburishabh/PRCV_Projects/Lane_detector_Using_OpenCV_for_autonomous_vehicle/include/LaneDetector.hpp"

using namespace cv;
using namespace std;

/*
* Takes 1 inputs
* Input is the path to the video source
*/

/**
 *@brief Function main that runs the main algorithm of the lane detection.
 *@brief It will read a video of a car in the highway and it will output the
 *@brief same video but with the plotted detected lane
 *@param argv[] is a string to the full path of the demo video
 *@return flag_plot tells if the demo has sucessfully finished
 */
int main(int argc, char *argv[]) {
    cout<<"starting...."<<endl;
    cout<<"Hello User"<<endl;
    cout<<"Please Choose from these options:\n"
      "1. Press 'g' = Denoise the image\n"
      "2. Press 'e' = Detect Edges\n"
      "3. Press 'm' = Mask Frame\n"
      "4. Press 'h' = Hough Transform\n"
      "5. Press 'f' = Final Output\n"
      "6. Press 's' = Save the image\n"
      "7. Press 'q' = quit\n"<<endl;
    if (argc != 2) {
      std::cout << "Not enough parameters" << std::endl;
      return -1;
    }

    // The input argument is the location of the video
    std::string source = argv[1];
    cv::VideoCapture* capdev;
    capdev = new cv::VideoCapture(source);
    if (!capdev -> isOpened()){
      cout<<"Unable to open Video Device\n"<<endl;
      return -1;
    }
    
    //get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "Expected size: " << refS.width << " " << refS.height << endl;

    //identifies a window
    cv::namedWindow("Video", cv::WindowFlags::WINDOW_NORMAL);
    

    LaneDetector lanedetector;  // Create the class object
    cv::Mat frame;
    cv::Mat img_denoise;
    cv::Mat img_edges;
    cv::Mat img_mask;
    cv::Mat img_lines;
    std::vector<cv::Vec4i> lines;
    std::vector<std::vector<cv::Vec4i> > left_right_lines;
    std::vector<cv::Point> lane;
    std::string turn;
    int flag_plot = -1;
    int i = 0;

    // Main algorithm starts. Iterate through every frame of the video
    while (i < 540) {
      *capdev >> frame;
      // Capture frame
      if (frame.empty()){
        cout<<"This frame is empty"<<endl;
        break;
      }
      static bool gPressed = false;
      static bool ePressed = false;
      static bool mPressed = false;
      static bool hPressed = false;
      static bool fPressed = false;

      bool anyKeyPressed = false;
      bool sPressed = false;

      //waits for a key press for 30 milliseconds
      char key = cv::waitKey(30);
      if(key == 'q'){
        cout<<"q is pressed"<<endl;
        break;
      }
      if(key == 's'){
        imwrite("Output_image.jpg", frame);
        cout<<"Image is saved"<<endl;
        sPressed = true;
      }
      if(key == 'g'){
        cout<<"g is pressed"<<endl;
        gPressed = true;
        ePressed = false;
        mPressed = false;
        hPressed = false;
        fPressed = false;
      }
      if(key == 'e'){
        cout<<"e is pressed"<<endl;
        gPressed = false;
        ePressed = true;
        mPressed = false;
        hPressed = false;
        fPressed = false;
      }
      if(key == 'm'){
        cout<<"m is pressed"<<endl;
        gPressed = false;
        ePressed = false;
        mPressed = true;
        hPressed = false;
        fPressed = false;
      }
      if(key == 'h'){
        cout<<"h is pressed"<<endl;
        gPressed = false;
        ePressed = false;
        mPressed = false;
        hPressed = true;
        fPressed = false;
      }
      if(key == 'f'){
        cout<<"f is pressed"<<endl;
        gPressed = false;
        ePressed = false;
        mPressed = false;
        hPressed = false;
        fPressed = true;
      }

      if(gPressed){
        anyKeyPressed = true;
        // Denoise the image using a Gaussian filter
        img_denoise = lanedetector.deNoise(frame);
        cv::imshow("Output Video", img_denoise);
        if(sPressed){
          cv::imwrite("Gaussian_Blur.png", img_denoise);
        }
      }
      
      if(ePressed){
        anyKeyPressed = true;
        // Detect edges in the image
        img_denoise = lanedetector.deNoise(frame);
        img_edges = lanedetector.edgeDetector(img_denoise);
        cv::imshow("Output Video", img_edges);
        if(sPressed){
          cv::imwrite("Edge_detection.png", img_edges);
        }
      }

      if(mPressed){
        anyKeyPressed = true;
        img_denoise = lanedetector.deNoise(frame);
        img_edges = lanedetector.edgeDetector(img_denoise);
        // Mask the image so that we only get the ROI
        img_mask = lanedetector.mask(img_edges);
        cv::imshow("Output Video", img_mask);
        if(sPressed){
          cv::imwrite("Masked_frame.png", img_mask);
        }
      }

      if(hPressed){
        anyKeyPressed = true;
        img_denoise = lanedetector.deNoise(frame);
        img_edges = lanedetector.edgeDetector(img_denoise);
        // Mask the image so that we only get the ROI
        img_mask = lanedetector.mask(img_edges);
        // Obtain Hough lines in the cropped image
        lines = lanedetector.houghLines(img_mask);
        // Draw the lines on a new image
        img_lines = cv::Mat::zeros(img_mask.size(), CV_8UC3);
        for (const auto& line : lines) {
          cv::line(img_lines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        }
        // Display the image with the lines
        cv::imshow("Output Video", img_lines);
        if (sPressed) {
          cv::imwrite("Hough_transform.png", img_lines);
        }
      }

      if(fPressed){
        anyKeyPressed = true;
        img_denoise = lanedetector.deNoise(frame);
        img_edges = lanedetector.edgeDetector(img_denoise);
        // Mask the image so that we only get the ROI
        img_mask = lanedetector.mask(img_edges);
        // Obtain Hough lines in the cropped image
        lines = lanedetector.houghLines(img_mask);
        // Draw the lines on a new image
        img_lines = cv::Mat::zeros(img_mask.size(), CV_8UC3);
        for (const auto& line : lines) {
          cv::line(img_lines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        }
        if (!img_lines.empty()) {
        // Separate lines into left and right lines
        left_right_lines = lanedetector.lineSeparation(lines, img_edges);

        // Apply regression to obtain only one line for each side of the lane
        lane = lanedetector.regression(left_right_lines, frame);

        // Predict the turn by determining the vanishing point of the the lines
        turn = lanedetector.predictTurn();

        // Plot lane detection
        flag_plot = lanedetector.plotLane(frame, lane, turn);

        i += 1;
        cv::waitKey(25);
      } 
      else {
        flag_plot = -1;
      }
    }
  }
  delete capdev;
  return flag_plot;
}
