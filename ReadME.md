# Lane Detector Using OpenCV for Autonomous Vehicle
Autonomous Driving Car is one of the most disruptive innovations in AI. One of the many steps involved during the training of an autonomous driving car is lane detection. Road Lane Detection requires to detection of the path of self-driving cars and avoiding the risk of entering other lanes. Lane recognition algorithms reliably identify the location and borders of the lanes by analyzing the visual input. Advanced driver assistance systems (ADAS) and autonomous vehicle systems both heavily rely on them. Today we will be talking about one of these lane detection algorithms.
	
## Steps to Implement Road Lane Detection

	<img src = "images/Pipeline.png" width=1000px>
	
- **Remove Noise :**
   1. Apply Gaussian filter mask to make the image smoother.
   2. Remove unwanted pixel values that could interfere with lane detection.

- **Edge Detection :**
   1. Convert the image to grayscale.
   2. Convert the grayscale image to binary.
   3. Use a special pattern to find edges by applying a row kernel to each pixel.

- **Region of Interest :**
   1. Here we create a mask for the Edge detected image to isolate the region of interest (ROI) containing the lane lines. 
   2. It creates a binary polygon mask using four points (pts) and fills it with blue color. 
   3. The mask is then applied to the edges image using a bitwise AND operation, and 
   4. The resulting masked image is returned as output.

- **Hough Lines Detection :**
   1. This function performs the Hough transform on the img_mask image to detect line segments that may correspond to the lane boundaries. 
   2. It uses the Probabilistic Hough Transform (cv::HoughLinesP()) and specifies the minimum line length, maximum gap between segments, and other parameters. 
   3. The detected lines are returned as a vector of cv::Vec4i objects.

- **Line Classification :**
   1. This function separates the detected lines into left and right lines based on their slopes and positions. 
   2. It iterates through the lines vector, calculates the slope of each line segment, and checks if it meets certain conditions. 
   3. Lines with slopes greater than a threshold and located on the appropriate sides of the image center are classified as right lines, 
   4. while lines with negative slopes and located on the other side are classified as left lines. 
   5. The resulting separated lines are returned as a vector of two vectors: output[0] contains the right lines, and output[1] contains the left lines.

- **Regression :**
   1. The regression function is defined within the LaneDetector class. 
   2. It takes two parameters: left_right_lines, which is a vector of vectors of cv::Vec4i representing the separated lines for the left and right lanes, and 	  inputImage, which is the original input image. It returns a vector of cv::Point that represents the line segments forming the lane boundaries.
    
   3. If right lines are being detected, fit a line using all the init and final points of the lines
   4. This code block checks if the right_flag is set to true, indicating that right lines have been detected. 
   5. If so, it iterates through each line in left_right_lines[0], which represents the right lane lines. 
   6. The initial and final points of each line are extracted and added to the right_pts vector. 
   7. After collecting the points, the cv::fitLine function is used to fit a line to the points using the least squares method. 
   8. The resulting line is stored in right_line, and the slope (right_m) and y-intercept (right_b) of the line are calculated.
  
   9. One the slope and offset points have been obtained, apply the line equation to obtain the line points
  10. Here, the code calculates the y-coordinates (ini_y and fin_y) for the starting and ending points of the lane lines
  11. Based on the image dimensions and a predefined value of 470. Using the slopes (right_m and left_m), y-intercepts (right_b and left_b), and the   		calculated y-coordinates, the x-coordinates of the lane lines are determined using the line equation. 
  12. Finally, the four points representing the lane boundaries are stored in the output vector.
  

- **Turn Prediction :**
  1. This function predicts the turn direction based on the vanishing point of the lane boundaries. 
  2. It calculates the x-coordinate of the vanishing point (vanish_x) using the slopes and intercepts of the right and left lines. 
  3. Depending on the position of the vanishing point relative to the image center, it determines if the road is turning left, turning right, or going  		straight. 
  4. The predicted turn direction is returned as a string (output).
   
- **Results :**
  1. This function plots the detected lane boundaries, turn prediction message, and a transparent polygon covering the area inside the lane boundaries. 
  2. It takes the original input image inputImage, the lane points lane, and the turn prediction turn.
  3. First, it creates a copy of the input image as output. Then, it creates a polygon using the lane points (poly_points) and fills it with red color. 
  4. The filled polygon is blended with the output image using cv::addWeighted(), creating a transparent effect.
  5. Next, it draws the left and right lane lines using yellow color (cv::line()). It also puts the turn prediction text on the image using cv::putText().
  6. Finally, it displays the final output image in a window named "Lane" using cv::imshow(). The function returns 0 to indicate success.

