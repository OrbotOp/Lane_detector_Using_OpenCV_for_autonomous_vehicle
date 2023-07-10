#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "/home/roburishabh/PRCV_Projects/Lane_detector_Using_OpenCV_for_autonomous_vehicle/include/LaneDetector.hpp"

// IMAGE BLURRING
/*
* Apply a Gaussian blur to the input image inputImage to reduce noise. 
* The blurred image is stored in output and then returned.
*/
cv::Mat LaneDetector::Remove_noise(cv::Mat inputImage) {
  cv::Mat output;

  cv::GaussianBlur(inputImage, output, cv::Size(3, 3), 0, 0);

  return output;
}

// EDGE DETECTION
/*
* Detect edges in the img_noise image using the Canny edge detection algorithm. 
* It converts the image to grayscale
* Apply a binary threshold to create a binary image, and 
* then uses a 1D kernel [-1 0 1] to compute the horizontal gradients. 
* The resulting edges are returned as output.
*/
cv::Mat LaneDetector::edgeDetector(cv::Mat img_noise) {
  cv::Mat output;
  cv::Mat kernel;
  cv::Point anchor;

  // Convert image from RGB to gray
  cv::cvtColor(img_noise, output, cv::COLOR_RGB2GRAY);
  // sets pixel values below the threshold value of 140 to 0 and 
  // pixel values above 140 or equal to the threshold to 255. 
  // This effectively converts the grayscale image into a binary image, 
  // where pixels representing edges are set to white (255) and other pixels to black (0).
  cv::threshold(output, output, 140, 255, cv::THRESH_BINARY);

  // Create the kernel [-1 0 1]
  // This kernel is based on the one found in the
  // Lane Departure Warning System by Mathworks
  anchor = cv::Point(-1, -1);
  kernel = cv::Mat(1, 3, CV_32F);
  kernel.at<float>(0, 0) = -1;
  kernel.at<float>(0, 1) = 0;
  kernel.at<float>(0, 2) = 1;

  // Filter the binary image to obtain the edges
  cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT);

  return output;
}

// MASK THE EDGE IMAGE
/*
* This function creates a mask for the img_edges image to isolate the region of interest (ROI) containing the lane lines. 
* It creates a binary polygon mask using four points (pts) and fills it with blue color. 
* The mask is then applied to the edges image using a bitwise AND operation, and 
* the resulting masked image is returned as output.
*/
cv::Mat LaneDetector::mask(cv::Mat img_edges) {
  cv::Mat output;
  // The mask matrix is initialized as a zero matrix with the same size and type as img_edges. 
  cv::Mat mask = cv::Mat::zeros(img_edges.size(), img_edges.type());
  // Additionally, an array of cv::Point called pts is defined, which represents the vertices of a polygonal mask to be applied.
  cv::Point pts[4] = {
      cv::Point(210, 720),
      cv::Point(550, 450),
      cv::Point(717, 450),
      cv::Point(1280, 720)
  };

  // This line fills the mask matrix with a convex polygon defined by the pts array. 
  // The cv::fillConvexPoly function takes the mask matrix, 
  // the array of points pts, the number of points (4 in this case), 
  // and the color cv::Scalar(255, 0, 0) (which represents blue) to fill the polygon with the specified color.
  cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0, 0));
  // img_edges: This is the input image matrix containing the edges or binary values that we want to mask.
  // mask: This is the mask matrix, which represents the region or shape that we want to apply as a mask. 
  //       The mask matrix should have the same size as the input image and contain binary values.
  // output: This is the output matrix where the result of the bitwise AND operation will be stored.

  // The cv::bitwise_and operation applies the AND operation on corresponding pixels of img_edges and mask matrices.
  // If both pixels have non-zero values (white), the resulting pixel in the output matrix will be set to non-zero. 
  // Otherwise, if any of the pixels is zero (black), the resulting pixel in the output matrix will be set to zero. 

  cv::bitwise_and(img_edges, mask, output);
  // The result is stored in the output matrix.
  return output;
}

// HOUGH LINES
/*
* This function performs the Hough transform on the img_mask image to detect line segments that may correspond to the lane boundaries. 
* It uses the Probabilistic Hough Transform (cv::HoughLinesP()) and specifies the minimum line length, maximum gap between segments, and other parameters. 
* The detected lines are returned as a vector of cv::Vec4i objects.
*/
std::vector<cv::Vec4i> LaneDetector::houghLines(cv::Mat img_mask) {
  std::vector<cv::Vec4i> line;

  // img_mask: This is the input binary image or edge-detected image where lines are to be detected.
  // line: This is the output vector to store the detected lines. Each line is represented as a cv::Vec4i vector, 
  //       where each element contains four values (x1, y1, x2, y2) representing the coordinates of the start and end points of a line segment.

  // The parameters passed to the cv::HoughLinesP function are as follows:
  // 1: The rho resolution of the accumulator in pixels. It defines the distance resolution of the accumulator.
  // CV_PI/180: The theta resolution of the accumulator in radians. It defines the angular resolution of the accumulator.
  // 20: The minimum number of votes or intersections required to consider a line.
  // 20: The minimum length of a line segment in pixels.
  // 30: The maximum allowed gap between line segments to link them together into a single line.
  HoughLinesP(img_mask, line, 1, CV_PI/180, 20, 20, 30);

  return line;
}

// SORT RIGHT AND LEFT LINES
/*
* This function separates the detected lines into left and right lines based on their slopes and positions. 
* It iterates through the lines vector, calculates the slope of each line segment, and checks if it meets certain conditions. 
* Lines with slopes greater than a threshold and located on the appropriate sides of the image center are classified as right lines, 
* while lines with negative slopes and located on the other side are classified as left lines. 
* The resulting separated lines are returned as a vector of two vectors: output[0] contains the right lines, and output[1] contains the left lines.
*/


// The lineSeparation() is defined within the LaneDetector class. 
// It takes two parameters: a vector of lines (lines) represented as cv::Vec4i, and an edge-detected image (img_edges). 
// It returns a vector of vectors of cv::Vec4i (output) representing the separated lines (right and left lanes). 
//  Additional variables and vectors are declared for further processing.
std::vector<std::vector<cv::Vec4i> > LaneDetector::lineSeparation(std::vector<cv::Vec4i> lines, cv::Mat img_edges) {
  std::vector<std::vector<cv::Vec4i> > output(2);
  size_t j = 0;
  cv::Point ini;
  cv::Point fini;
  double slope_thresh = 0.3;
  std::vector<double> slopes;
  std::vector<cv::Vec4i> selected_lines;
  std::vector<cv::Vec4i> right_lines, left_lines;

  // In this loop, each line from the input lines vector is processed. 
  // The start and end points of the line segment are extracted into ini and fini points, respectively. 
  for (auto i : lines) {
    ini = cv::Point(i[0], i[1]);
    fini = cv::Point(i[2], i[3]);
    // Basic algebra: slope = (y1 - y0)/(x1 - x0)
    double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))/(static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

    // If the absolute value of the slope is greater than slope_thresh, the slope and the line segment itself are added to the slopes and selected_lines vectors, respectively.
    // If the slope is too horizontal, discard the line
    // If not, save them  and their respective slope
    if (std::abs(slope) > slope_thresh) {
      slopes.push_back(slope);
      selected_lines.push_back(i);
    }
  }

  // Here, the separation of lines into left and right lanes is performed. 
  // The variable img_center is set to the x-coordinate of the center of the input img_edges image. 
  // A while loop is used to iterate through the selected_lines vector. 
  // For each line, the start and end points are extracted into ini and fini points, respectively.
  img_center = static_cast<double>((img_edges.cols / 2));
  while (j < selected_lines.size()) {
    ini = cv::Point(selected_lines[j][0], selected_lines[j][1]);
    fini = cv::Point(selected_lines[j][2], selected_lines[j][3]);

    // Condition to classify line as left side or right side
    // If the slope of the line (slopes[j]) is positive and both the end and start points have x-coordinates 
    // greater than the image center (img_center), the line is considered part of the right lane and is added to the right_lines vector. 
    // Similarly, if the slope is negative and both the end and start points have x-coordinates less than the image center, 
    // the line is considered part of the left lane and is added to the left_lines vector. 
    // The right_flag and left_flag variables are also updated accordingly.
    if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
      right_lines.push_back(selected_lines[j]);
      right_flag = true;
    } else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
        left_lines.push_back(selected_lines[j]);
        left_flag = true;
    }
    j++;
  }

  output[0] = right_lines;
  output[1] = left_lines;

  return output;
}

// REGRESSION FOR LEFT AND RIGHT LINES
/*
* The regression function is defined within the LaneDetector class. 
* It takes two parameters: left_right_lines, which is a vector of vectors of cv::Vec4i representing the separated lines for the left and right lanes,
* and inputImage, which is the original input image. It returns a vector of cv::Point that represents the line segments forming the lane boundaries. 
* Variables and vectors are initialized for further processing.
*/
std::vector<cv::Point> LaneDetector::regression(std::vector<std::vector<cv::Vec4i> > left_right_lines, cv::Mat inputImage) {
  std::vector<cv::Point> output(4);
  cv::Point ini;
  cv::Point fini;
  cv::Point ini2;
  cv::Point fini2;
  cv::Vec4d right_line;
  cv::Vec4d left_line;
  std::vector<cv::Point> right_pts;
  std::vector<cv::Point> left_pts;

  // If right lines are being detected, fit a line using all the init and final points of the lines
  // This code block checks if the right_flag is set to true, indicating that right lines have been detected. 
  // If so, it iterates through each line in left_right_lines[0], which represents the right lane lines. 
  // The initial and final points of each line are extracted and added to the right_pts vector. 
  // After collecting the points, the cv::fitLine function is used to fit a line to the points using the least squares method. 
  // The resulting line is stored in right_line, and the slope (right_m) and y-intercept (right_b) of the line are calculated.
  if (right_flag == true) {
    for (auto i : left_right_lines[0]) {
      ini = cv::Point(i[0], i[1]);
      fini = cv::Point(i[2], i[3]);

      right_pts.push_back(ini);
      right_pts.push_back(fini);
    }

    if (right_pts.size() > 0) {
      // The right line is formed here
      cv::fitLine(right_pts, right_line, cv::DIST_L2, 0, 0.01, 0.01);
      right_m = right_line[1] / right_line[0];
      right_b = cv::Point(right_line[2], right_line[3]);
    }
  }

  // If left lines are being detected, fit a line using all the init and final points of the lines
  // If the left_flag is set to true, indicating that left lines have been detected. 
  // If so, it iterates through each line in left_right_lines[1], which represents the left lane lines. 
  // The initial and final points of each line are extracted and added to the left_pts vector.
  // After collecting the points, the cv::fitLine() is used to fit a line to the points, and the resulting line is stored in left_line. 
  // The slope (left_m) and y-intercept (left_b) of the line are calculated.
  if (left_flag == true) {
    for (auto j : left_right_lines[1]) {
      ini2 = cv::Point(j[0], j[1]);
      fini2 = cv::Point(j[2], j[3]);

      left_pts.push_back(ini2);
      left_pts.push_back(fini2);
    }

    if (left_pts.size() > 0) {
      // The left line is formed here
      cv::fitLine(left_pts, left_line, cv::DIST_L2, 0, 0.01, 0.01);
      left_m = left_line[1] / left_line[0];
      left_b = cv::Point(left_line[2], left_line[3]);
    }
  }

  // One the slope and offset points have been obtained, apply the line equation to obtain the line points
  // Here, the code calculates the y-coordinates (ini_y and fin_y) for the starting and ending points of the lane lines
  // based on the image dimensions and a predefined value of 470. Using the slopes (right_m and left_m), y-intercepts (right_b and left_b), 
  // and the calculated y-coordinates, the x-coordinates of the lane lines are determined using the line equation. 
  // Finally, the four points representing the lane boundaries are stored in the output vector.
  int ini_y = inputImage.rows;
  int fin_y = 470;

  double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
  double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

  double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
  double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

  output[0] = cv::Point(right_ini_x, ini_y);
  output[1] = cv::Point(right_fin_x, fin_y);
  output[2] = cv::Point(left_ini_x, ini_y);
  output[3] = cv::Point(left_fin_x, fin_y);

  return output;
}

// TURN PREDICTION
/*
* This function predicts the turn direction based on the vanishing point of the lane boundaries. 
* It calculates the x-coordinate of the vanishing point (vanish_x) using the slopes and intercepts of the right and left lines. 
* Depending on the position of the vanishing point relative to the image center, it determines if the road is turning left, turning right, or going straight. 
* The predicted turn direction is returned as a string (output).
*/
std::string LaneDetector::predictTurn() {
  std::string output;
  double vanish_x;
  double thr_vp = 10;

  // The vanishing point is the point where both lane boundary lines intersect
  // This line calculates the x-coordinate of the vanishing point. 
  // It uses the slopes (right_m and left_m) and y-intercepts (right_b and left_b) of the fitted lines for the right and left lanes. 
  // The formula (right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y represents the intersection of the two lines in the x-axis. 
  // By dividing this by the difference between the slopes (right_m - left_m), the x-coordinate of the vanishing point is obtained.
  vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

  // The vanishing points location determines where is the road turning
  // checks the value of vanish_x to determine the predicted turn based on the x-coordinate of the vanishing point. 
  // If vanish_x is less than the image center minus the threshold thr_vp, 
  // it means the vanishing point is significantly to the left of the center, indicating a left turn. 
  // If vanish_x is greater than the image center plus thr_vp, it means the vanishing point is significantly to the right of the center, 
  // indicating a right turn. Otherwise, if vanish_x is within the range of the image center plus/minus thr_vp, it means the vanishing point is close to the center, indicating a straight path.
  if (vanish_x < (img_center - thr_vp))
    output = "Left Turn";
  else if (vanish_x > (img_center + thr_vp))
    output = "Right Turn";
  else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
    output = "Straight";

  return output;
}

// PLOT RESULTS
/*
* This function plots the detected lane boundaries, turn prediction message, and a transparent polygon covering the area inside the lane boundaries. 
* It takes the original input image inputImage, the lane points lane, and the turn prediction turn.
* First, it creates a copy of the input image as output. Then, it creates a polygon using the lane points (poly_points) and fills it with red color. 
* The filled polygon is blended with the output image using cv::addWeighted(), creating a transparent effect.
* Next, it draws the left and right lane lines using yellow color (cv::line()). It also puts the turn prediction text on the image using cv::putText().
* Finally, it displays the final output image in a window named "Lane" using cv::imshow(). The function returns 0 to indicate success.
*/
int LaneDetector::plotLane(cv::Mat inputImage, std::vector<cv::Point> lane, std::string turn) {
  std::vector<cv::Point> poly_points;
  cv::Mat output;

  // Create the transparent polygon for a better visualization of the lane
  inputImage.copyTo(output);
  poly_points.push_back(lane[2]);
  poly_points.push_back(lane[0]);
  poly_points.push_back(lane[1]);
  poly_points.push_back(lane[3]);
  cv::fillConvexPoly(output, poly_points, cv::Scalar(0, 0, 255), cv::LINE_AA, 0);
  cv::addWeighted(output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage);

  // Plot both lines of the lane boundary
  cv::line(inputImage, lane[0], lane[1], cv::Scalar(0, 255, 255), 5, cv::LINE_AA);
  cv::line(inputImage, lane[2], lane[3], cv::Scalar(0, 255, 255), 5, cv::LINE_AA);

  // Plot the turn message
  cv::putText(inputImage, turn, cv::Point(50, 90), cv::FONT_HERSHEY_COMPLEX_SMALL, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

  // Show the final output image
  cv::namedWindow("Lane", cv::WindowFlags::WINDOW_NORMAL);
  cv::imshow("Lane", inputImage);
  return 0;
}
