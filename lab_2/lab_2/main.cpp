#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/intensity_transform.hpp>
#include <chrono>



void boxFilterCustom(cv::Mat& src, cv::Mat& dst, int kernel_w, int kernel_h) {
    //Проверка ядра
    if (kernel_w == 1 || kernel_h == 1) {
        kernel_h = 3;
        kernel_w = 3;
    }
    else if (kernel_w % 2 == 0 || kernel_h % 2 == 0) {
        kernel_h = 3;
        kernel_w = 3;
    }

    //Создание ядра бокс фильтра
    cv::Mat kernel(kernel_h, kernel_w, CV_64FC1, cv::Scalar(1.0 / (kernel_h * kernel_w)));

    int pad_r = (kernel_h - 1) / 2;
    int pad_c = (kernel_w - 1) / 2;

    //Паддинг изображения нулями 
    cv::Mat pad_img(cv::Size(src.cols + pad_c * 2, src.rows + pad_r * 2), CV_64FC1, cv::Scalar(0));
    src.copyTo(pad_img(cv::Rect(pad_c, pad_r, src.cols, src.rows)));

    //Создание временного элемента для дальнейшей работы с ним
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_64FC1);

    //Проход по изображению
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            tmp.at<double>(i, j) = sum(kernel.mul(pad_img(cv::Rect(j, i, kernel_w, kernel_h)))).val[0];
        }
    }

    tmp.convertTo(dst, CV_8UC1);
    return;
}


void Compare_img(cv::Mat& src1, cv::Mat& src2) {
    // Создание элемента отражающего разницу между изображениями
    cv::Mat dif = cv::Mat::zeros(src1.rows, src1.cols, CV_8UC1);
    // Счетчик совпадающих пикселей
    int counter_sim = 0;
    // Указатели на массивы данных изображений
    uint8_t* p_1 = src1.data;
    uint8_t* p_2 = src2.data;
    // Итератор, тк все изображения имеют тот-же размер есть смысл использовать только один
    cv::MatIterator_<uint8_t> it, end;
    for (it = dif.begin<uint8_t>(), end = dif.end<uint8_t>(); it != end; it++) {
        if (*p_1++ == *p_2++ ) {
            ++counter_sim;
        } else {
            *it = 255;
        }
    }

    // Процент совпадения
    double sim = 100.0 * (double)counter_sim / (double)(src1.rows * src1.cols);

    // Вывод 
    std::cout << "the " << sim << "% of similarity" << std::endl;
    cv::imshow("Similarity", dif);
    return;
}

cv::Mat logTransform(cv::Mat& src) {
    cv::Mat lg_im;
    src.convertTo(lg_im, CV_32F);
    lg_im = lg_im + 1;
    cv::log(lg_im, lg_im);
    cv::convertScaleAbs(lg_im, lg_im);
    cv::normalize(lg_im, lg_im, 0, 255, cv::NORM_MINMAX);
    return lg_im;
}

cv::Mat unsharpMask(cv::Mat& src, cv::Mat& filtr, int sharp) {
    cv::Mat dif;
    cv::Mat res = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    //Вычитание
    cv::subtract(src, filtr, dif);

    cv::MatIterator_<uint8_t> it, end;
    cv::MatIterator_<uint8_t> it_src = src.begin<uint8_t>();
    cv::MatIterator_<uint8_t> it_dif = dif.begin<uint8_t>();
    for (it = res.begin<uint8_t>(), end = res.end<uint8_t>(); it != end; it++, it_src++, it_dif++) {
        int buf = (int)(*it_src) + sharp * (int)(*it_dif);
        *it = cv::saturate_cast<uint8_t>(buf);
    }
    return res;
}

void laplaseFilterCustom(cv::Mat& src, cv::Mat& dst) {
    //Создание ядра бокс фильтра
    int kernel_h = 3;
    int kernel_w = 3;
    cv::Mat kernel = (cv::Mat_<double>(kernel_h, kernel_w) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

    int pad_r = (kernel_h - 1) / 2;
    int pad_c = (kernel_w - 1) / 2;

    //Паддинг изображения нулями 
    cv::Mat pad_img(cv::Size(src.cols + pad_c * 2, src.rows + pad_r * 2), CV_64FC1, cv::Scalar(0));
    src.copyTo(pad_img(cv::Rect(pad_c, pad_r, src.cols, src.rows)));

    //Создание временного элемента для дальнейшей работы с ним
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_64FC1);

    //Проход по изображению
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            tmp.at<double>(i, j) = sum(kernel.mul(pad_img(cv::Rect(j, i, kernel_w, kernel_h)))).val[0];
        }
    }

    tmp.convertTo(dst, CV_8UC1);
    return;
}

int main()
{
    cv::Mat src = cv::imread("lena.png");
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    int k_size = 3;

    cv::Mat res_1;
    cv::Mat res_2;
    
    //--------------------------test 1-3-----------------------------------------------
    auto t_1 = std::chrono::high_resolution_clock::now();
    cv::blur(src, res_2, cv::Size(k_size, k_size));
    auto t_2 = std::chrono::high_resolution_clock::now();
    boxFilterCustom(src, res_1, k_size, k_size);
    auto t_3 = std::chrono::high_resolution_clock::now();

    auto t_cust = std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2);
    std::cout << "Custom : " << t_cust.count() << " mics" << std::endl;

    auto t_b_in = std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1);
    std::cout << "Built-in : " << t_b_in.count() << " mics" << std::endl;

    imshow("Source", src);
    imshow("BoxCust", res_1);
    imshow("BoxB_in", res_2);
    Compare_img(res_1, res_2);
    cv::waitKey(0);
    //-------------------------------------------------------------------------------
    //--------------------------test 4-----------------------------------------------
    cv::Mat dif;
    cv::blur(src, res_2, cv::Size(k_size, k_size));
    cv::GaussianBlur(src, res_1, cv::Size(k_size, k_size), 1);
    //Разность изображений
    cv::subtract(res_1, res_2, dif);
    //Логарифмическая фильтрация ( пришлось писать самому)
    //cv::intensity_transform::logTransform(dif, dif);
   
    imshow("Source", src);
    imshow("Gaussian", res_1);
    imshow("Box", res_2);
    imshow("Dif", dif);
    imshow("Dif_l", logTransform(dif));
    cv::waitKey(0);
    //---------------------------------------------------------------------------------
    //--------------------------test 5-------------------------------------------------
    int sharp = 3;
    cv::Mat res_3; 
    laplaseFilterCustom(src, res_3);
    imshow("laplase", res_3);
    cv::Mat unsh_b = unsharpMask(src, res_2, sharp);
    cv::Mat unsh_g = unsharpMask(src, res_1, sharp);
    cv::Mat unsh_l = unsharpMask(src, res_3, sharp);

    //Разность изображений
    cv::subtract(unsh_b, unsh_g, dif);

    imshow("USP_B", unsh_b);
    imshow("USP_G", unsh_g);
    imshow("USP_L", unsh_l);
    imshow("Dif", dif);
    imshow("Dif_l", logTransform(dif));
    cv::waitKey(0);
    //---------------------------------------------------------------------------------
    //--------------------------test 6 - 7---------------------------------------------
    /*cv::Mat unsh_b = unsharpMask(src, res_2, sharp);
    laplaseFilterCustom(src, res_1);
    imshow("Source", src);
    imshow("res", res_1);
    cv::waitKey(0);*/
    //---------------------------------------------------------------------------------
    return 0;
}