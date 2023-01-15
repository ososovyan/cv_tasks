#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

const double PI = 3.14;


void wheel() {
    cv::Mat img_b = cv::imread("background.jpg");
    //cv::Mat img_b = cv::imread("background.jpg");
    cv::Size size_background = cv::Size(2000, 1200);
    cv::resize(img_b, img_b, size_background, cv::INTER_LINEAR);
    cv::Size s_obj = cv::Size(50, 50);
    cv::Size trace = cv::Size(10, 10);
    cv::Point init_pose = cv::Point(60, size_background.height / 2);
    cv::Point res_pose = init_pose;

    int x_speed = 1;
    int y_speed = 0;
    int time = 0;
    cv::Mat tmp_img;

    while (1) {
        img_b.copyTo(tmp_img);
        y_speed = 6 * cos(double(time) / double(100));
        res_pose.x = res_pose.x + x_speed;
        res_pose.y = res_pose.y + y_speed;
        cv::Point rect_1 = cv::Point(res_pose.x - s_obj.height, res_pose.y - s_obj.width);
        cv::Point rect_2 = cv::Point(res_pose.x + s_obj.height, res_pose.y + s_obj.width);

        cv::Point tr_1 = cv::Point(res_pose.x - trace.height, res_pose.y - trace.width);
        cv::Point tr_2 = cv::Point(res_pose.x + trace.height, res_pose.y + trace.width);
        rectangle(img_b, tr_1, tr_2, cv::Scalar(0, 0, 0), cv::FILLED);
        rectangle(tmp_img, rect_1, rect_2, cv::Scalar(0, 0, 0), 5);
        if (res_pose.x == size_background.width / 2) {
            cv::imwrite("res.jpg", tmp_img);
        }

        
        cv::imshow("task_2", tmp_img);
        cv::waitKey(2);
        time++;
        if (res_pose.x == size_background.width - s_obj.width) {
            cv::destroyAllWindows();
            break;
        }
    }

}

cv::Mat blending(cv::Mat& img_1, cv::Mat& img_2, double alph) {
    //проверка на масштаб
    cv::Mat tmp_1 = img_1.clone();
    cv::Mat tmp_2 = img_2.clone();
    if (tmp_1.size() != tmp_2.size()) {
        cv::resize(tmp_2, tmp_2, tmp_1.size(), cv::INTER_LINEAR);
    }

    cv::Mat tmp(tmp_1.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    //Верность коэффициента альфа
    if (!(alph >= 0 && alph <= 1)) {
        return tmp;
    }

    // Итератор, тк все изображения имеют тот-же размер есть смысл использовать только один
    cv::MatIterator_<cv::Vec3b> it, end, it_1, it_2;
    it_1 = tmp_1.begin<cv::Vec3b>();
    it_2 = tmp_2.begin<cv::Vec3b>();
    double buf[3];
    for (it = tmp.begin<cv::Vec3b>(), end = tmp.end<cv::Vec3b>(); it != end; it++, it_1++, it_2++) {
        buf[0] = alph * (double)((*it_1)[0]) + (1 - alph) * (double)(*it_2)[0];
        buf[1] = alph * (double)((*it_1)[1]) + (1 - alph) * (double)(*it_2)[1];
        buf[2] = alph * (double)((*it_1)[2]) + (1 - alph) * (double)(*it_2)[2];
        (*it)[0] = cv::saturate_cast<uint8_t>(buf[0]);
        (*it)[1] = cv::saturate_cast<uint8_t>(buf[1]);
        (*it)[2] = cv::saturate_cast<uint8_t>(buf[2]);
    }
    return tmp;
}

int main() {
    //--------------------------------------extra task-------------------------------
    cv::Mat img1(cv::Size(256, 256), CV_8UC3, cv::Scalar(0, 0, 256));
    cv::Mat img2(cv::Size(256, 256), CV_8UC3, cv::Scalar(0, 256, 0));
    double a = 0.1;
    cv::imshow("1", img1);
    cv::imshow("2", img2);
    cv::imshow("extra", blending(img1, img2, 0.5));
    cv::waitKey(0);
    cv::destroyAllWindows();
    //--------------------------------------------------------------------------------
    
    wheel();
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
