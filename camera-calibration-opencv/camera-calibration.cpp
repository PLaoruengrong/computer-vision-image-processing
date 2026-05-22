#include <iostream>		// console input/output
#include <fstream>		// file reading/writing
#include <filesystem>

//include all necessary headers
#include <opencv2/core/core.hpp>		//core functionality	
#include <opencv2/highgui/highgui.hpp>	// needed to display images on screen
#include <opencv2/calib3d/calib3d.hpp>	// all functionalities for camera calibration
#include <opencv2/imgproc/imgproc.hpp>  // needed to convert colored images into grayscale images

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
	//Parameters;
	Size2i	image_size	= Size2i(640,480);
	Size2i	board_size	= Size2i(9,6);
	Size2f	square_size = Size2f(22.0f, 21.5f);

	// Adjust Input folder and Output folder
	string  input_folder = "camera_calibration_lab/Images";
	string  output_folder = "camera_calibration_lab/Output";

	//Initialize all necessary variables
	vector<Point2f>				corners;								// vector of 2D float Points (stores the corner points of one image)
	vector<vector<Point2f>>		image_Points;							// vector of several vectors of 2D float Points (stores the corner points of all images)
	vector<vector<Point3f>>		object_Points;							// vector of several vectors of 3D float Points (stores the corner points of all images in 3D)

	vector<Mat>	image_Seq;
	vector<string> image_names;												// vector of Mat, which holds all original images

	Mat	intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); 		// camera calibration Matrix
	Mat	distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));		// Mat storing the 5 distortion coefficients k1,k2,p1,p2,k3

	vector<cv::Mat> rotation_vectors;									// vector of Rotation matrices for all images
	vector<cv::Mat> translation_vectors;								// vector of tranlation vectors for all images


	// ---------------------------------------------------------------------------------- 
	//
	// Load images from folder, find and store the chessboard corners for each image.
	//
	// ---------------------------------------------------------------------------------- 
	for (const auto & entry : filesystem::directory_iterator(input_folder))
	{
		string imageFileName = entry.path().filename();
		image_names.push_back(imageFileName);
		Mat image = imread(input_folder + imageFileName);
		    if (image.empty())
				{
					cout << "Failed to load image:" << imageFileName << endl;
					return -1;
				}
		
		Mat imageGray;
		// find chessboard corners
		bool patternfound = findChessboardCorners(imageGray,board_size,corners);
		if (!patternfound || corners.size()!= board_size.width*board_size.height)
		{
			cout << "Can not find chessboard corners in image " << imageFileName  << ", please take image out of dataset!\n";
			exit(1);
		}
		else
		{
	        cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY); // convert color image to grayscale
	        bool patternfound = findChessboardCorners(imageGray,board_size,corners); // extract intersection points of checkerboard pattern
            
            // ADAPTIVE_THRESH: use adaptive threshold to convert color to BW in dealing with uneven lighting or shadows, NORMALIZE_IMAGE: normalize brightness and contrast (reduce noise for vignett)
            bool patternfound = findChessboardCorners(imageGray, board_size, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            
            // Add subpixel interpolation
            cv::cornerSubPix(imageGray,corners,cv::Size(11, 11),cv::Size(-1, -1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,30,0.001));

		}

        drawChessboardCorners(image, board_size, corners, patternfound); // draw corner
        imshow("Checkerboard Corners Found", image);
        waitKey(500);
            
		image_Points.push_back(corners);
		image_Seq.push_back(image);
	}


	// ---------------------------------------------------------------------------------- 
	//
	// Initialize the 3D coordinates of the corners in the chessboard coordinate system
	//
	// ----------------------------------------------------------------------------------   
	for (int i = 0; i < image_names.size() ; i++)
	{
		vector<Point3f> tempPointSet;
		for (int k = 0; k<board_size.height; k++)
		{
			for (int l = 0; l<board_size.width; l++)
			{
				// Assume that the chessboard is located on the x-y-plane     
				// The origin of the world coordinate is the top left corner of the chessboard       
				Point3f tempPoint;
				tempPoint.x = l*square_size.width; // coordinate = the index * the size of that side
				tempPoint.y = k*square_size.height;
				tempPoint.z = 0.0f; // zero z-coordinate
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}

	// ---------------------------------------------------------------------------------- 
	//
	// Do the actual calibration
	//
	// ----------------------------------------------------------------------------------   

	// Perform calibration
	double rms_error = calibrateCamera(object_Points, image_Points, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors);
	
	// Change the parameters
	double rms_error = calibrateCamera(object_Points, image_Points, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, CALIB_FIX_PRINCIPAL_POINT | CALIB_ZERO_TANGENT_DIST);
	
	cout << "Calibration completed! \n";

	// ---------------------------------------------------------------------------------- 
	//
	// Let's do some Evalutation
	//
	// ----------------------------------------------------------------------------------   
	cout << "Starting evaluation of the results:" << endl;
	double total_err = 0.0;                       /* Sum of the mean error over pictures */
	double err = 0.0;                             /* Mean error of each picture   */


	cout << "Calibration error of each picture:" << endl;
	for (int i = 0; i < image_names.size() ; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];

		
		// Compute the theoretical projection of 3D points with the obtained intrinsic and extrinstric parameters of the camera  
		vector<Point2f>  image_points2;               // theoretical projections
		
		// Calculate projection of points
        cv::projectPoints(tempPointSet, rotation_vectors[i],translation_vectors[i], intrinsic_matrix, distortion_coeffs,image_points2);

		// compute the error between new calculated projection points and the measured points */
        err = cv::norm(image_Points[i], image_points2, cv::NORM_L2) / tempPointSet.size(); // mean reprojection error = average total accumulated error over all points
        total_err += err;
	}
	cout << "Mean error of all pictures:" << total_err / image_names.size() << "pixel" << endl;
	cout << "Evaluation done!" << endl;


	// ---------------------------------------------------------------------------------- 
	//
	// Save the calibration result 
	//
	// ----------------------------------------------------------------------------------   

	ofstream fout(output_folder + "CalibrationResult.txt");  
	
	cout << "Starting saving calibration results" << endl;
	
	fout << "Intrinstic Matrix:" << endl;
	fout << intrinsic_matrix << endl;
	fout << "" << endl;
	fout << "Distortion coefficients:\n";
	fout << distortion_coeffs << endl;
	fout << "" << endl;
	fout << "Mean error of all pictures:" << total_err / image_names.size() << "pixel" << endl << endl;
	fout << "" << endl;

	cout << "Saving completed" << endl;

	// ---------------------------------------------------------------------------------- 
	//
	// Save the undistorted images 
	//
	// ----------------------------------------------------------------------------------   

	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "Saving calibrated pictures" << endl;
	for (int i = 0; i < image_names.size() ; i++)
	{
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		Mat undistortedImage = image_Seq[i].clone();
		remap(image_Seq[i], undistortedImage, mapx, mapy, INTER_CUBIC);
		string imageFilename =  "undistorted_";
		imageFilename += image_names[i];
		imwrite(output_folder + imageFilename, undistortedImage);
	}


	cout << "DONE" << endl;
	return 0;
}
