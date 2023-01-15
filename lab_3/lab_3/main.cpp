#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
 //��� ����� �������
typedef enum {
	RED_ROB,
	GREEN_ROB,
	BLUE_ROB,
} OBJ_TYPE;

void task_1(cv::Mat& src) {
	cv::Mat tmp = src.clone();

	//�������������� � ������������� ����������� �����������
	cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
	//���������
	GaussianBlur(tmp, tmp, cv::Size(5, 5), 2);
	//�����������
    threshold(tmp, tmp, 220, 255, cv::THRESH_BINARY);
	//���������� �������
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	//������ ��� �������� ������ � �� ����
	if (contours.empty()) {
		return;
	}
	//���� �������� ����� ( ������ 5) �� �������� ��� ������ �� �����, 
	//�� ��������� ������ ���� ����������� ��� �������
	if (contours.size() > 2) {
		return;
	}
	//���� ����� ������� ������ �������� ��� ����
	int target = 0;
	double buf = cv::contourArea(contours[target]);
	for (int i = 0; i != contours.size(); ++i) {
		if (cv::contourArea(contours[i]) > buf) {
			buf = cv::contourArea(contours[i]);
			target = i;
		}
	}
	//��� � ������
	cv::drawContours(src, contours, target, cv::Scalar(255, 0, 255), 2);
	// ����� �������� � ������ �������� �������
	cv::Moments m = cv::moments(contours[target]);
	//����� ������ ����,� ��� ����������� �� �����
	double cm_x = m.m10 / m.m00;
	cv::line(src, cv::Point(cm_x, 0), cv::Point(cm_x, src.rows), cv::Scalar(155, 234, 90), 1);
	double cm_y = m.m01 / m.m00;
	cv::line(src, cv::Point(0, cm_y), cv::Point(src.cols, cm_y), cv::Scalar(155, 234, 90), 1);

}
void task_2(cv::Mat& src) {
	cv::Mat tmp = src.clone();

	//�������������� � ������ ����� ������� ������
	cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
	//���������
	GaussianBlur(tmp, tmp, cv::Size(3, 3), 2);
	cv::inRange(tmp, cv::Scalar(0, 0, 0), cv::Scalar(40, 255, 255), tmp);
	//���������� �������
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	//������ ��� �������� ������ � �� ����
	if (contours.empty()) {
		return;
	}
	//���� ����� ������� ������ �������� ��� ����
	int target = 0;
	double buf = cv::contourArea(contours[target]);
	for (int i = 0; i != contours.size(); ++i) {
		if (cv::contourArea(contours[i]) > buf) {
			buf = cv::contourArea(contours[i]);
			target = i;
		}
	}
	//��� � ������
	cv::drawContours(src, contours, target, cv::Scalar(255, 0, 255), 2);
	// ����� �������� � ������ �������� �������
	cv::Moments m = cv::moments(contours[target]);
	//����� ������ ����,� ��� ����������� 
	double cm_x = m.m10 / m.m00;
	cv::line(src, cv::Point(cm_x, 0), cv::Point(cm_x, src.rows), cv::Scalar(155, 234, 90), 1);
	double cm_y = m.m01 / m.m00;
	cv::line(src, cv::Point(0, cm_y), cv::Point(src.cols, cm_y), cv::Scalar(155, 234, 90), 1);
	imshow("src", src);
	cv::waitKey(0);
}


void findRobotContour(cv::Mat& src, cv::Mat& dst, OBJ_TYPE obj, cv::Point cm_l) {
	cv::Mat tmp = src.clone();
	//������ � ������ ���� �� �����, ��� �� ��������� ������
	cv::ellipse(tmp, cv::Point(640, 217), cv::Size(32, 25), 0, 0, 360, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
	
	cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
	//�������������� ������� �������� �������
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Scalar color;
	//� �������� ����� ��� ���������
	cv::Mat red_lower = src.clone();
	cv::Mat red_upper = src.clone();
	//���������� �� ������� ���������(��� ������� ����� ����)
	switch (obj) {
	case RED_ROB:
		cv::inRange(tmp, cv::Scalar(0, 40, 59), cv::Scalar(10, 168, 255), red_lower);
		cv::inRange(tmp, cv::Scalar(170, 0, 0), cv::Scalar(180, 255, 255), red_upper);
		cv::bitwise_or(red_lower, red_upper, tmp);
		color = cv::Scalar(0, 0, 255);
		break;
	case GREEN_ROB:
		cv::inRange(tmp, cv::Scalar(59, 50, 0), cv::Scalar(77, 255, 255), tmp);
		color = cv::Scalar(0, 255, 0);
		break;
	case BLUE_ROB:
		cv::inRange(tmp, cv::Scalar(80, 50, 0), cv::Scalar(100, 255, 255), tmp);
		color = cv::Scalar(255, 0, 0);
		break;
	default:
		break;
	}


	//������ ���� ������ � �������
	cv::erode(tmp, tmp, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
	cv::dilate(tmp, tmp, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));

	cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	//������ ���� ������ � �������, ������� ���������� ������
	//������ ��� �������� ������ � �� ����
	if (contours.empty()) {
		return;
	}
	int target = 0;
	//�d����� ����� curernt, ���� ������� ������
	double buf_l = 0;
	//�d����� ����� global min
	double buf_l_min = 0;
	
	for (int i = 0; i != contours.size(); ++i) {
		
		if (cv::contourArea(contours[i]) > 150 ) {
			cv::drawContours(dst, contours, i, color, 3);
			cv::Moments m = cv::moments(contours[i]);
			//����� ������ ����,� ��� ��������� 
			double cm_x = m.m10 / m.m00 ;
			double cm_y = m.m01 / m.m00;
			//��������� � ���� �������� ��� �����
			cm_x -= cm_l.x;
			cm_y -= cm_l.y;
			buf_l = cm_x * cm_x + cm_y * cm_y;
			//��� ��� ������� ����� � ��� ���� ����. �� ������ �������� ����� ���������������� 
			if (i == 0) {
				buf_l_min = buf_l;
			}
			else if (buf_l_min > buf_l) {
				//���� ������ �� ����� ���
				buf_l_min = buf_l;
				target = i;
			}
		}
	}
	//���������� ������ ������
	cv::Moments m = cv::moments(contours[target]);
	double cm_x = m.m10 / m.m00;
	double cm_y = m.m01 / m.m00;
	cv::circle(dst, cv::Point(cm_x, cm_y), 4, color, -1);
	return ;
}

void findLightContour(cv::Mat& src, cv::Point& cm_l) {
	cv::Mat tmp = src.clone();
	//�������������� � ������������� ����������� �����������
	cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
	//���������
	GaussianBlur(tmp, tmp, cv::Size(5, 5), 2);
	//�����������
	threshold(tmp, tmp, 250, 255, cv::THRESH_BINARY);
	//���������� �������
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(tmp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	//������ ��� �������� ������ � �� ����
	if (contours.empty()) {
		return;
	}
	//���� ����� ������� ������ �������� ��� ����(������� �������� �������)
	int target = 0;
	double buf = cv::contourArea(contours[target]);
	for (int i = 0; i != contours.size(); ++i) {
		if (cv::contourArea(contours[i]) > buf) {
			buf = cv::contourArea(contours[i]);
			target = i;
		}
	}
	//��� � ������
	//cv::drawContours(src, contours, target, cv::Scalar(255, 0, 255), 2);
	// ����� �������� � ������ �������� �������
	cv::Moments m = cv::moments(contours[target]);
	//����� ������ ����,� ��� ����������� �� �����
	double cm_x = m.m10 / m.m00;
	double cm_y = m.m01 / m.m00;
	cv::circle(src, cv::Point(cm_x, cm_y), 4, cv::Scalar(0, 0, 0), -1);
	cm_l = cv::Point(cm_x, cm_y);
	return;
}


void gk(cv::Mat& img, cv::Mat& temp) {

	cv::Mat tmp_img = img.clone();
	cv::Mat tmp_temp = temp.clone();

	//������������� � ���������� �������� ��� �����������
	cv::cvtColor(tmp_img, tmp_img, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(tmp_img, tmp_img, cv::Size(3, 3), 2);
	cv::threshold(tmp_img, tmp_img, 240, 255, cv::THRESH_BINARY_INV);
	
	std::vector<std::vector<cv::Point>> contours_img;
	std::vector<cv::Vec4i> hierarchy_img;

	cv::findContours(tmp_img, contours_img, hierarchy_img, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//������������� ��� �������, ��������
	cv::cvtColor(tmp_temp, tmp_temp, cv::COLOR_BGR2GRAY);
	cv::threshold(tmp_temp, tmp_temp, 150, 255, cv::THRESH_BINARY);
	
	std::vector<std::vector<cv::Point>>  contours_temp;
	std::vector<cv::Vec4i>  hierarchy_temp;

	cv::findContours(tmp_temp, contours_temp, hierarchy_temp, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//�������� ��������
	for (int i = 0; i != contours_img.size(); ++i) {
		if (cv::matchShapes(contours_img[i], contours_temp[0], cv::CONTOURS_MATCH_I2, 0) > 0.9) {
			cv::drawContours(img, contours_img, i, cv::Scalar(0, 0, 255), 4);
		}
		else {
			cv::drawContours(img, contours_img, i, cv::Scalar(0, 255, 0), 4);
		}
	}
}
int main() {
	//����� ���������������� ������ ����� 
	/*//-----------------------------test_1 video---------------------------------------------------
	std::string path_1 = "img_zadan/allababah/V2.mp4";
	cv::VideoCapture cap(path_1);
	cv::Mat frame ;
	while (true) {
		cap.read(frame);
		task_1(frame);
		imshow("Image", frame);
		cv::waitKey(20);
	}
	*///----------------------------------------------------------------------------------------
	/*//-----------------------------test_2---------------------------------------------------
	std::string path_2 = "img_zadan/teplovizor/size0-army.mil-2008-08-28-082221.jpg";
	cv::Mat img = cv::imread(path_2);
	task_2(img);

	
	*///----------------------------------------------------------------------------------------
	/*//-----------------------------test_3 img---------------------------------------------------
	std::string path_3 = "img_zadan/roboti/roi_robotov_1.jpg";
	cv::Mat src = cv::imread(path_3);
	cv::Mat tmp = src.clone();
	cv::Point cm_l;
	findLightContour(src, cm_l);
	findRobotContour(tmp, src, RED_ROB, cm_l);
	findRobotContour(tmp, src, GREEN_ROB, cm_l);
	findRobotContour(tmp, src, BLUE_ROB, cm_l);
	imshow("res", src);
	cv::waitKey(0);
	*///----------------------------------------------------------------------------------------
	/*//---------------------------- - test_3 video-------------------------------------------------- 
	std::string path_4 = "img_zadan/roboti/v_rob.mp4";
	cv::VideoCapture cap(path_4);
	cv::Mat frame;
	while (true) {
		cap.read(frame);
		
		cv::Mat tmp = frame.clone();
		cv::Point cm_l;
		findLightContour(frame, cm_l);
		findRobotContour(tmp, frame, RED_ROB, cm_l);
		findRobotContour(tmp, frame, GREEN_ROB, cm_l);
		findRobotContour(tmp, frame, BLUE_ROB, cm_l);
		//cv::circle(src, cv::Point(640, 220), 26, cv::Scalar(0, 0, 0), -1);

		imshow("Video", frame);
		cv::waitKey(20);
	}
	*///----------------------------------------------------------------------------------------
	//-----------------------------test_4 img---------------------------------------------------
	std::string path_41 = "img_zadan/gk/gk.jpg";
	cv::Mat img = cv::imread(path_41);

	std::string path_42 = "img_zadan/gk/gk_tmplt.jpg";
	cv::Mat temp = cv::imread(path_42);

	gk(img, temp);
	imshow("img", img);
	cv::waitKey(0);
	//----------------------------------------------------------------------------------------

	return 0;
}