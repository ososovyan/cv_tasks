#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include "aruco_samples_utility.hpp"
#include <ctime>

static bool readDetectorParameters(std::string filename, cv::aruco::DetectorParameters& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
        
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params.cornerRefinementMethod;

    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;

    return true;
}

cv::Ptr<cv::aruco::Board> create_board() {
    int markers_x = 7;
    int markers_y = 4;
    int marker_size = 90;
    int separation_size = 90;
    int margins = separation_size;
    int dict_id = cv::aruco::DICT_6X6_50;
    int border_bits = 1;

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::Dictionary::get(dict_id);

    cv::Size image_size ;
    image_size.width = markers_x * (marker_size + separation_size) - separation_size + 2 * margins;
    image_size.height = markers_y * (marker_size + separation_size) - separation_size + 2 * margins;

    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(markers_x, markers_y, float(marker_size), float(separation_size), dictionary);
    cv::Mat board_image;
    cv::aruco::drawPlanarBoard(board, image_size, board_image, margins, border_bits);
    cv::imwrite("board.png", board_image);

    cv::imshow("board", board_image);
    return board;
    cv::waitKey(0);

}
void create_marker() {
    cv::Mat marker_image;
    cv::Ptr<cv::aruco::Dictionary> dict_id = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::aruco::drawMarker(dict_id, 20, 300, marker_image, 1);
    cv::imwrite("marker.png", marker_image);
    cv::imshow("marker_image", marker_image);
    cv::waitKey(0);
}
enum flags { fix_point, zero_tg_dist, fix_ratio };
void calibrateCamera(flags flag) {
    std::string output_file = "calibration.xml"; 
    cv::String params_filename = "params.yml"; 
    int markers_x = 7;
    int markers_y = 4;
    int marker_size = 90;
    int separation_size = 90;
    int margins = separation_size;
    int dict_id = cv::aruco::DICT_6X6_50;
    int border_bits = 1;

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::Dictionary::get(dict_id);

    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(markers_x, markers_y, float(marker_size), float(separation_size), dictionary);

    const float aspect_ratio = 1.0;
    int callibration_flags = 0;

    if (fix_ratio) {    
        callibration_flags |= cv::CALIB_FIX_ASPECT_RATIO;
    }
    if (zero_tg_dist) {
        callibration_flags |= cv::CALIB_ZERO_TANGENT_DIST;
    }
    if (fix_point) {
        callibration_flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    }
    cv::Ptr<cv::aruco::DetectorParameters> detector_ptr;
    cv::aruco::DetectorParameters detector_params;

    readDetectorParameters(params_filename, detector_params);
    detector_ptr = &detector_params;
    bool refindStrategy = true; 

    
    
    //cv::VideoCapture input(0);

    cv::VideoCapture input("video.mp4");
    input.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    input.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    const int waitTime = 5;
    if (!input.isOpened()) {
        std::cerr << "Video hasn't been captured" << std::endl;
    }

    std::vector<std::vector<std::vector<cv::Point2f>>> all_corners; 
    std::vector<std::vector<int>> all_ids;      
    cv::Size image_size = cv::Size(1920, 1080);

    while (input.grab()) {
        cv::Mat image, imageCopy;
        input.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;

        cv::aruco::detectMarkers(image, dictionary, corners, ids, detector_ptr, rejected);

        
        if (refindStrategy) {
            cv::aruco::refineDetectedMarkers(image, board, corners, ids, rejected);
        }
        
        image.copyTo(imageCopy);
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }
        putText(imageCopy, "Press ENTER to add current frame. 'ESC' to finish and calibrate",
            cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

        imshow("out", imageCopy);
        char key = (char)cv::waitKey(waitTime);
        if (key == 27) break;
        if (key == 13 && ids.size() > 0) {
            std::cout << "Frame captured" << std::endl;
            all_corners.push_back(corners);
            all_ids.push_back(ids);
            image_size = image.size();
        }
    }
    input.release();
    cv::destroyAllWindows();
    if (all_ids.size() < 1) {       
        std::cerr << "Not enough captures for calibration" << std::endl;
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;   
    std::vector<cv::Mat> rvecs, tvecs;
    double reprojection_error; 

    
    std::vector<std::vector<cv::Point2f>> all_corners_concatenated;
    std::vector<int> all_ids_concatenated;
    std::vector<int> marker_counter_per_frame;
    marker_counter_per_frame.reserve(all_corners.size());
    for (unsigned int i = 0; i < all_corners.size(); i++) {
        marker_counter_per_frame.push_back((int)all_corners[i].size());
        for (unsigned int j = 0; j < all_corners[i].size(); j++) {
            all_corners_concatenated.push_back(all_corners[i][j]);
            all_ids_concatenated.push_back(all_ids[i][j]);
        }
    }
    
    reprojection_error = cv::aruco::calibrateCameraAruco(all_corners_concatenated, all_ids_concatenated,
        marker_counter_per_frame, board, image_size, cameraMatrix,
        distCoeffs, rvecs, tvecs, callibration_flags);
    
    bool saveOk = saveCameraParams(output_file, image_size, aspect_ratio, flag, cameraMatrix, distCoeffs, reprojection_error);
    if (!saveOk) {
        std::cerr << "Cannot save output file" << std::endl;
        return;
    }
 
    std::cout << "Rep Error: " << reprojection_error << std::endl;
    //std::cout << "Calibration saved to " << output_file << std::endl;
    return;
}

void create_cubes(cv::Mat& image, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    const int ms = 90;
    std::vector<cv::Scalar> colours{ cv::Scalar(255, 0, 0),
                                     cv::Scalar(0, 255, 0), 
                                     cv::Scalar(0, 0, 255), 
                                     cv::Scalar(0, 0, 0),
                                     cv::Scalar(255, 255, 255),
                                     cv::Scalar(255, 0, 255),
                                     cv::Scalar(0, 255, 255),
                                     cv::Scalar(255, 255, 0),
                                     cv::Scalar(120, 120, 120),
                                     cv::Scalar(120, 120, 0),
                                     cv::Scalar(120, 0, 0),
                                     cv::Scalar(0, 120, 0)};
    std::vector<cv::Point3d> cube_sc_points;   
    cube_sc_points.push_back(cv::Point3d(ms / 2, ms / 2, 0));
    cube_sc_points.push_back(cv::Point3d(ms / 2, -ms / 2, 0));
    cube_sc_points.push_back(cv::Point3d(-ms / 2, -ms / 2, 0));
    cube_sc_points.push_back(cv::Point3d(-ms / 2, ms / 2, 0));
    cube_sc_points.push_back(cv::Point3d(ms / 2, ms / 2, ms));
    cube_sc_points.push_back(cv::Point3d(ms / 2, -ms / 2, ms));
    cube_sc_points.push_back(cv::Point3d(-ms / 2, -ms / 2, ms));
    cube_sc_points.push_back(cv::Point3d(-ms / 2, ms / 2, ms));
    std::vector<cv::Point2d> image_sc_points;
    for (size_t i = 0; i < tvecs.size(); i++) {
       
        cv::projectPoints(cube_sc_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs, image_sc_points);
        for (size_t i = 0; i < image_sc_points.size(); i++) {
            cv::line(image, image_sc_points[0], image_sc_points[1], colours[0], 3);
            cv::line(image, image_sc_points[0], image_sc_points[3], colours[1], 3);
            cv::line(image, image_sc_points[0], image_sc_points[4], colours[2], 3);
            cv::line(image, image_sc_points[1], image_sc_points[2], colours[3], 3);
            cv::line(image, image_sc_points[1], image_sc_points[5], colours[4], 3);
            cv::line(image, image_sc_points[2], image_sc_points[3], colours[5], 3);
            cv::line(image, image_sc_points[2], image_sc_points[6], colours[6], 3);
            cv::line(image, image_sc_points[3], image_sc_points[7], colours[7], 3);
            cv::line(image, image_sc_points[4], image_sc_points[5], colours[8], 3);
            cv::line(image, image_sc_points[4], image_sc_points[7], colours[9], 3);
            cv::line(image, image_sc_points[6], image_sc_points[7], colours[10],3);
            cv::line(image, image_sc_points[6], image_sc_points[5], colours[11],3);
        }
    }
}


void detection() {
    cv::VideoCapture video("video.mp4");
    
    cv::String filename = "params.yml";
    cv::String cam_filename = "calibration.xml";
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::aruco::DetectorParameters detector_params;
    cv::Ptr<cv::aruco::DetectorParameters> detector_ptr;
    readDetectorParameters(filename, detector_params);
    detector_ptr = &detector_params;

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::VideoWriter output("video_1.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        video.get(cv::CAP_PROP_FPS),
        cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH),
            video.get(cv::CAP_PROP_FRAME_HEIGHT)));

    if (!output.isOpened())
    {
        std::cout << "!!! Output video could not be opened" << std::endl;
        return;
    }
    readCameraParameters(cam_filename, camera_matrix, dist_coeffs);
    while (video.grab()) {  
        cv::Mat image, image_tmp;
        video.retrieve(image);
        image.copyTo(image_tmp);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        cv::aruco::detectMarkers(image_tmp, dictionary, corners, ids, detector_ptr, rejected);
        if (ids.size() > 0) {
            cv::aruco::estimatePoseSingleMarkers(corners, 90, camera_matrix, dist_coeffs, rvecs, tvecs); 
            create_cubes(image_tmp, rvecs, tvecs, camera_matrix, dist_coeffs);    
        }
        cv::imshow("out", image_tmp);
        char key = (char)cv::waitKey(10);
        output.write(image_tmp);
        if (key == 27) {
            break;
        }
    }
}


int main() {
    //create_board();
    //create_marker();
    //calibrateCamera(fix_ratio);
    //create_board();
    detection();
	cv::waitKey(0);

	return 0;
}