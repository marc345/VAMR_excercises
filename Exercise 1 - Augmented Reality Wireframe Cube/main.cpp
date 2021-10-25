#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main()
{   
    const uint8_t board_rows = 5;
    const uint8_t board_cols = 8;

    int corners[board_rows+1][board_cols+1][3] = {0};
    
    // world coordinates of the checkerboard corners
    for(uint8_t i=0; i<board_rows+1; i++){
        for(uint8_t j=0; j<board_cols+1; j++){
            corners[i][j][0] = j*4;  // world coordinate x position in cm
            corners[i][j][1] = i*4;  // world coordinate y position in cm
            corners[i][j][2] = 0;    // world coordinate z position in cm
            // std::cout << "corners[][][] = " \
                // << corners[i][j][0] << ", " \
                // << corners[i][j][1] << ", " \
                // << corners[i][j][2] \
                // << '\n'; 
        }
    }



cv::Mat image1;

image1=cv::imread("./data/images/img_0001.jpg");
//reads the input image
cv::namedWindow("Checkerboard",cv::WINDOW_AUTOSIZE);
cv::cvtColor(image1, image1, cv::COLOR_RGB2GRAY, 1);
cv::imshow("Checkerboard",image1);

cv::waitKey(0);

cv::destroyWindow("Checkerboard");

return 0;
}

