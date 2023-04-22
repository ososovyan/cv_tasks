#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
void krasivSpektr(cv::Mat& magI) {
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void show_magnitude(cv::Mat& complex_img, std::string name, bool isReverse = false) {
    cv::Mat planes[] = { cv::Mat_<float>(complex_img), cv::Mat::zeros(complex_img.size(), CV_32F) };
    cv::split(complex_img, planes);
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_i = planes[0];

    mag_i += cv::Scalar::all(1);

    if (!isReverse) {
        log(mag_i, mag_i);

        mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
        krasivSpektr(mag_i);
    }
    else {
        mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
        //log(mag_i, mag_i);
        //krasivSpektr(mag_i);
    }


    cv::normalize(mag_i, mag_i, 0, 1, cv::NORM_MINMAX);

    cv::imshow( name + "spectrum magnitude - " , mag_i);

}
cv::Mat_<cv::Complex<float>> to_complex(cv::Mat& src) {
    cv::Mat_<cv::Complex<float>> output(src.rows, src.cols);
    for (int i = 0; i <  src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            output.at<cv::Complex<float>>(i, j).re = src.at<cv::Vec2f>(i, j)[0];
            output.at<cv::Complex<float>>(i, j).im = src.at<cv::Vec2f>(i, j)[1];
        }
    }
    return output;
}

cv::Mat from_complex(cv::Mat_<cv::Complex<float>>& src) {
    cv::Mat output(src.rows, src.cols, CV_32FC2);
    for (int i = 0; i <  output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.at<cv::Vec2f>(i, j)[0] = src.at<cv::Complex<float>>(i, j).re;
            output.at<cv::Vec2f>(i, j)[1] = src.at<cv::Complex<float>>(i, j).im;
        }
    }
    return output;
}

cv::Mat_<cv::Complex<float>> create_euler_m(int size, bool isReverse = false) {
    cv::Mat euler_m = cv::Mat(size, size, CV_32FC2);
    int a = 0;
    if (isReverse) {
        a = -1;
    }
    else {
        a = 1;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float cell_el_power = a * (2 * CV_PI / size) * i * j;
            euler_m.at<cv::Complex<float>>(i, j).re = cos(cell_el_power);
            euler_m.at<cv::Complex<float>>(i, j).im = sin(cell_el_power);
        }
    }
    return euler_m;
}

cv::Mat_<cv::Vec2f> my_DFT(cv::Mat src, bool isReverse) {
    cv::Mat_ < cv::Complex<float >> comlex_img = to_complex(src);

    cv::Mat_<cv::Complex<float>> euler_m_c = create_euler_m(src.cols, isReverse);
    cv::Mat_<cv::Complex<float>> euler_m_r = create_euler_m(src.rows, isReverse);

    cv::Mat_<cv::Complex<float>> tmp = euler_m_r * comlex_img * euler_m_c;

    cv::Mat_<cv::Vec2f>output = from_complex(tmp);

    return output;
}


void cooley_tukey(cv::Mat_<cv::Complex<float>>& src, bool isReverse) {
    if (src.cols <= 1) {
        return;
    }

    cv::Mat_<cv::Complex<float>> odd(1, src.cols / 2);
    cv::Mat_<cv::Complex<float>> even(1, src.cols / 2);
    for (int i = 0; i != src.cols / 2; ++i) {
        even.at<cv::Complex<float>>(0, i) = src.at<cv::Complex<float>>(0, 2 * i);
        odd.at<cv::Complex<float>>(0, i) = src.at<cv::Complex<float>>(0, 2 * i + 1);
    }

    cooley_tukey(odd, isReverse);
    cooley_tukey(even, isReverse);

    int a = 0;
    if (isReverse) {
        a = 1;
    }
    else {
        a = -1;
    }

    for (int j = 0; j < src.cols / 2; j++) {
        float e_power = a * (2 * CV_PI / src.cols) * j;
        cv::Complex<float> W(cos(e_power), sin(e_power)) ;

        src.at<cv::Complex<float>>(0, j) = even.at<cv::Complex<float>>(0, j) + W * odd.at<cv::Complex<float>>(0, j);
        src.at<cv::Complex<float>>(0, src.cols/2 + j) = even.at<cv::Complex<float>>(0, j) - W * odd.at<cv::Complex<float>>(0, j);
    }
}

cv::Mat_<cv::Complex<float>> fft(cv::Mat_<cv::Complex<float>>& src, bool isReverse = false) {
    for (int i = 0; i != src.rows; ++i) {
        cv::Mat_<cv::Complex<float>> row = src.row(i);
        cooley_tukey(row, isReverse);
        src.row(i) = row;
    }
    cv::transpose(src, src);

    for (int i = 0; i != src.rows; ++i) {
        cv::Mat_<cv::Complex<float>> row = src.row(i);
        cooley_tukey(row, isReverse);
        src.row(i) = row;
    }
    cv::transpose(src, src);
    return src;
}

void build_in_dft(cv::Mat& src){
    cv::Mat padded;                            
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols); 
    
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    auto start_b_i = std::chrono::high_resolution_clock::now();
    cv::dft(complexI, complexI);
    auto end_b_i = std::chrono::high_resolution_clock::now();
    auto duration_my = std::chrono::duration_cast<std::chrono::microseconds>(end_b_i - start_b_i);
    std::cout << "b-in DFT time " << duration_my.count() << " mks" << "\n";

    cv::Mat inv_conv_img;

    
    cv::dft(complexI, inv_conv_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    


    show_magnitude(inv_conv_img, "b_in_back" ,true);

    cv::split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_i = planes[0];

    mag_i += cv::Scalar::all(1);                    
    log(mag_i, mag_i);

    mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
    krasivSpektr(mag_i);

    cv::normalize(mag_i, mag_i, 0, 1, cv::NORM_MINMAX);

    cv::imshow("b_in_Input Image", src);    
    cv::imshow("b_in_spectrum magnitude", mag_i);
    cv::waitKey();
}

void my_own_dft(cv::Mat& src, bool isReverse = false) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);
    
    complex_img = to_complex(complex_img);
    
    auto start_m_o = std::chrono::high_resolution_clock::now();
    cv::Mat_<cv::Complex<float>> euler_m_c = create_euler_m(complex_img.cols);
    cv::Mat_<cv::Complex<float>> euler_m_r = create_euler_m(complex_img.rows);

    cv::Mat_<cv::Complex<float>> tmp = euler_m_r * complex_img * euler_m_c;
    auto end_m_o = std::chrono::high_resolution_clock::now();
    auto duration_m_o = std::chrono::duration_cast<std::chrono::microseconds>(end_m_o - start_m_o);
    std::cout << "m-o DFT time " << duration_m_o.count() << " mks" << "\n";
    complex_img = from_complex(tmp);

    cv::Mat_<cv::Complex<float>> euler_m_c_b = create_euler_m(tmp.cols, true);
    cv::Mat_<cv::Complex<float>> euler_m_r_b = create_euler_m(tmp.rows, true);

    cv::Mat_<cv::Complex<float>> tmp_back = euler_m_r_b * tmp * euler_m_c_b;
    show_magnitude(tmp_back, "my_o_back", true);

    cv::split(complex_img, planes);
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_i = planes[0];

    mag_i += cv::Scalar::all(1);
    log(mag_i, mag_i);

    mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
    krasivSpektr(mag_i);

    cv::normalize(mag_i, mag_i, 0, 1, cv::NORM_MINMAX);

    cv::imshow("my-o_Input Image", src);
    cv::imshow("my_o_spectrum magnitude", mag_i);
    cv::waitKey();
}


void my_own_fft(cv::Mat& src, bool isReverse = false) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    //std::cout << m << n << src.rows << src.cols <<std::endl;
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);

    cv::Mat_<cv::Complex<float>> tmp = to_complex(complex_img);
    auto start_fft = std::chrono::high_resolution_clock::now();
    tmp = fft(tmp, isReverse);
    auto end_fft = std::chrono::high_resolution_clock::now();
    auto duration_fft = std::chrono::duration_cast<std::chrono::microseconds>(end_fft - start_fft);
    std::cout << " FFT time " << duration_fft.count() << " mks" << "\n";
    complex_img = from_complex(tmp);

    tmp = fft(tmp, true);
    show_magnitude(tmp, "but", true);

    cv::split(complex_img, planes);
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_i = planes[0];

    mag_i += cv::Scalar::all(1);
    log(mag_i, mag_i);

    mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
    krasivSpektr(mag_i);

    cv::normalize(mag_i, mag_i, 0, 1, cv::NORM_MINMAX);

    cv::imshow("butInput Image", src);
    cv::imshow("butspectrum magnitude", mag_i);
    cv::waitKey();
}


void inint(cv::Mat& src_1, cv::Mat& src_2, cv::Mat& complex_1, cv::Mat& complex_2) {
    cv::Mat padded_1;
    copyMakeBorder(src_1, padded_1, 0, src_2.rows, 0, src_2.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_1[] = { cv::Mat_<float>(padded_1), cv::Mat::zeros(padded_1.size(), CV_32F) };
    cv::merge(planes_1, 2, complex_1);

    cv::Mat padded_2;
    copyMakeBorder(src_2, padded_2, 0, src_1.rows, 0, src_1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_2[] = { cv::Mat_<float>(padded_2), cv::Mat::zeros(padded_2.size(), CV_32F) };
    cv::merge(planes_2, 2, complex_2);
}



void convolution(cv::Mat& src, cv::Mat& kernel, std::string kernel_name) {
    cv::Mat complex_img_1, complex_img_2;
    inint(src, kernel, complex_img_1, complex_img_2);

    cv::dft(complex_img_1, complex_img_1);
    cv::dft(complex_img_2, complex_img_2);

    cv::Mat conv_img;
    cv::mulSpectrums(complex_img_1, complex_img_2, conv_img, 0, false);

    cv::Mat inv_conv_img;
    cv::dft(conv_img, inv_conv_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    cv::Mat res = inv_conv_img(cv::Rect(1, 1, src.cols, src.rows)).clone();

    cv::imshow("src_img", src);
    std::cout << src.rows << src.cols << "\n" << std::endl;
    show_magnitude(complex_img_1, "image");
    show_magnitude(complex_img_2, "kernel - " + kernel_name);
    show_magnitude(conv_img, "coonv");
    show_magnitude(res, "res", true);
    std::cout << res.rows << res.cols << "\n" << std::endl;
    cv::waitKey();
    
}

void filtration(cv::Mat& src, bool isHigh = false){

    
    cv::Mat planes[] = { cv::Mat_<float>(src), cv::Mat::zeros(src.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    cv::dft(complexI, complexI);
    krasivSpektr(complexI);

    cv::Mat spectr = complexI.clone();
    krasivSpektr(spectr);
    //show_magnitude(spectr, "just");
    cv::Point centre(complexI.cols / 2, complexI.rows / 2);

    if (isHigh) {
        cv::circle(complexI, centre, 20, cv::Scalar::all(0), -1);
    }
    else {
        cv::Mat tmp = complexI.clone();
        cv::circle(tmp, centre, 60, cv::Scalar::all(0), -1);
        cv::bitwise_xor(tmp, complexI, complexI);
    }

    cv::Mat output;
    krasivSpektr(complexI);
    dft(complexI, output, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    output.convertTo(output, CV_8U);

    
    
    cv::Mat planes_[] = { cv::Mat_<float>(spectr), cv::Mat::zeros(spectr.size(), CV_32F) };
    cv::split(spectr, planes_);
    magnitude(planes_[0], planes_[1], planes_[0]);
    cv::Mat mag_i = planes_[0];

    mag_i += cv::Scalar::all(1);
    log(mag_i, mag_i);

    mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));
    krasivSpektr(mag_i);
  
    cv::normalize(mag_i, mag_i, 0, 1, cv::NORM_MINMAX);
    if (isHigh) {
        cv::circle(mag_i, centre, 20, cv::Scalar::all(0), -1);
    }
    else {
        cv::Mat tmp = mag_i.clone();
        cv::circle(tmp, centre, 60, cv::Scalar::all(0), -1);
        cv::bitwise_xor(tmp, mag_i, mag_i);
    }

    show_magnitude(spectr, "just");
    cv::imshow(" filtred spectrum magnitude - ", mag_i);
    cv::imshow("Input Image", src);
    cv::imshow("Output", output);
    cv::waitKey();

    

}


void correlation(cv::Mat& src_1, cv::Mat& src_2) {
    cv::Mat temp_src_1, temp_src_2;
    src_1.convertTo(temp_src_1, CV_64F);
    src_2.convertTo(temp_src_2, CV_64F);

    cv::subtract(temp_src_1, cv::mean(temp_src_1), temp_src_1);
    cv::subtract(temp_src_2, cv::mean(temp_src_2), temp_src_2);

    cv::Mat pad_src_1, pad_src_2;
    cv::copyMakeBorder(temp_src_1, pad_src_1, 0, temp_src_2.rows -1, 0, temp_src_2.cols -1, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    int n = cv::getOptimalDFTSize(pad_src_1.rows);;
    int m = cv::getOptimalDFTSize(pad_src_1.cols);;
    cv::copyMakeBorder(pad_src_1, pad_src_1, 0, n - pad_src_1.rows , 0, m - pad_src_1.cols , cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(temp_src_2, pad_src_2, 0, pad_src_1.rows - temp_src_2.rows, 0, pad_src_1.cols - temp_src_2.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes_1[] = { cv::Mat_<double>(pad_src_1), cv::Mat::zeros(pad_src_1.size(), CV_64F) };
    cv::Mat complex_src_1;
    cv::merge(planes_1, 2, complex_src_1);

    cv::Mat planes_2[] = { cv::Mat_<double>(pad_src_2), cv::Mat::zeros(pad_src_2.size(), CV_64F) };
    cv::Mat complex_src_2;
    cv::merge(planes_2, 2, complex_src_2);

    cv::dft(complex_src_1, complex_src_1);
    cv::dft(complex_src_2, complex_src_2);

    cv::Mat conv_res, conv_res_back;
    cv::mulSpectrums(complex_src_1, complex_src_2, conv_res, 0, true);
    cv::dft(conv_res, conv_res_back, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat conv_res_back_crop = conv_res_back(cv::Rect(0, 0, src_1.cols, src_1.rows)).clone();
    cv::normalize(conv_res_back_crop, conv_res_back_crop, 0, 1, cv::NORM_MINMAX);

    double min, max;
    cv::Mat output;
    cv::minMaxLoc(conv_res_back_crop, &min, &max);
    cv::threshold(conv_res_back_crop, output, max - 0.06, max, cv::THRESH_BINARY);
    cv::imshow("img", src_1);
    cv::imshow("temp", src_2);
    cv::imshow("Result", conv_res_back_crop);
    cv::imshow("Res", output);
    cv::waitKey(0);
}


int main() {
    cv::Mat img = cv::imread("250px-Fourier2.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(256, 256));

    cv::Mat img_1 = cv::imread("num3.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread("6.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat sobelV = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobelH = (cv::Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    cv::Mat box = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    cv::Mat lap = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

    build_in_dft(img);
    my_own_dft(img);
    my_own_fft(img);
   // convolution(img, box, "BOX");
    //convolution(img, sobelV, "sobelV");
   // convolution(img, sobelH, "sobelH");
    //convolution(img, lap, "lap");
    //filtration(img);
    //correlation(img_1, img_2);
    return 0;
}