# Lane Detector Using OpenCV for Autonomous Vehicle
Autonomous Driving Car is one of the most disruptive innovations in AI. One of the many steps involved during the training of an autonomous driving car is lane detection. Road Lane Detection requires to detection of the path of self-driving cars and avoiding the risk of entering other lanes. Lane recognition algorithms reliably identify the location and borders of the lanes by analyzing the visual input. Advanced driver assistance systems (ADAS) and autonomous vehicle systems both heavily rely on them. Today we will be talking about one of these lane detection algorithms.
	
## Steps to Implement Road Lane Detection
	<img src="images/Pipeline.png" width=1000px>
	
- Capturing and decoding video file: We will capture the video using Capdev and after the capturing has been initialized every video frame is decoded (i.e. converting into a sequence of images)

- Grayscale conversion of image: The video frames are in RGB format, RGB is converted to grayscale because processing a single channel image is faster than processing a three-channel colored image.

- Reduce noise: Noise can create false edges, therefore before going further, it’s imperative to perform image smoothening. Gaussian blur is used to perform this process. Gaussian blur is a typical image filtering technique for lowering noise and enhancing image characteristics. The weights are selected using a Gaussian distribution, and each pixel is subjected to a weighted average that considers the pixels surrounding it. By reducing high-frequency elements and improving overall image quality, this blurring technique creates softer, more visually pleasant images.

- Canny Edge Detector: It computes gradient in all directions of our blurred image and traces the edges with large changes in intensity. For more explanation please go through this article: Canny Edge Detector

- Region of Interest: This step is to take into account only the region covered by the road lane. A mask is created here, which is of the same dimension as our road image. Furthermore, bitwise AND operation is performed between each pixel of our canny image and this mask. It ultimately masks the canny image and shows the region of interest traced by the polygonal contour of the mask.

- Hough Line Transform: In image processing, the Hough transformation is a feature extraction method used to find basic geometric objects like lines and circles. By converting the picture space into a parameter space, it makes it possible to identify shapes by accumulating voting points. We’ll use the probabilistic Hough Line Transform in our algorithm. The Hough transformation has been extended to address the computational complexity with the probabilistic Hough transformation. In order to speed up processing while preserving accuracy in shape detection, it randomly chooses a selection of picture points and applies the Hough transformation solely to those points.

- Draw lines on the Image or Video: After identifying lane lines in our field of interest using Hough Line Transform, we overlay them on our visual input(video stream/image).


