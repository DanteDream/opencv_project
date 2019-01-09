#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv\cv.h>
#include<assert.h>

#include "opencv2/features2d/features2d.hpp" 
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <opencv2/ml/ml.hpp> 
#include<math.h>

using namespace std;
using namespace cv;

const char Input_Title[] = "Input Image";
const char Output_Title[] = "Output Image";

Mat DoSobel(Mat &img)
{
	Mat out;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(img, img, img.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);//x方向
	convertScaleAbs(img, out);

	return out;
}

Mat DoThreshold(Mat &img)
{
	Mat out;
	threshold(img, out, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

	return out;
}

Mat DoMorpholigy(Mat &img)
{
	Mat out;
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 5));
	morphologyEx(img, out, MORPH_CLOSE, element);

	return out;
}
void DoFindContours(Mat &img, Mat &out)
{
	RNG rng(1345);
	vector<vector<Point>> contourPoints;//一个数组
	vector<Vec4i> hierachy;


}


Mat erode_dilate_Pict(Mat img)
{
	Mat img_good, img_erode, img_dilate;
	Mat elementX = getStructuringElement(MORPH_RECT, Size(25, 1));
	Mat elementY = getStructuringElement(MORPH_RECT, Size(1, 19));//(1,19)
	Point point(-1, -1);
	//x方向膨胀腐蚀
	dilate(img, img_dilate, elementX, point, 3);//3
	erode(img_dilate, img_erode, elementX, point, 5);//5
	dilate(img_erode, img_dilate, elementX, point, 3);//3
	//y方向膨胀腐蚀
	//dilate(img_dilate, img_dilate, elementY, point, 1);//没有的
	erode(img_dilate, img_erode, elementY, point, 2);//2
	dilate(img_erode, img_dilate, elementY, point, 3);//3

	//噪声处理，用中值滤波
	medianBlur(img_dilate, img_good, 15);
	medianBlur(img_good, img_good, 15);


	return img_good;
}


Mat findCar(Mat img, Mat image)
{

	Mat dst;
	image.copyTo(dst, img);
	return dst;
}

int findPointXMax(vector<Point> vec)
{
	int max;
	int i = vec.size();
	cout << i;
	i = i - 1;
	max = vec[i].x;
	i = i - 1;
	while (i >= 0)
	{
		if (max < vec[i].x)
		{
			max = vec[i].x;
		}
		i--;
	}
	return max;
}

int findPointYMax(vector<Point> vec)
{
	int max;
	int i = vec.size();
	max = vec[i - 1].y;
	i = i - 2;
	while (i >= 0)
	{

		if (max < vec[i].y)
		{
			max = vec[i].y;
		}
		i--;
	}
	return max;
}

int findPointXMin(vector<Point> vec)
{
	int min;
	int i = vec.size();
	min = vec[i - 1].x;
	i = i - 2;
	while (i >= 0)
	{

		if (min > vec[i].x)
		{
			min = vec[i].x;
		}
		i--;
	}
	return min;
}


int findPointYMin(vector<Point> vec)
{
	int min;
	int i = vec.size();
	min = vec[i - 1].y;
	i = i - 2;
	while (i >= 0)
	{

		if (min > vec[i].y)
		{
			min = vec[i].y;
		}
		i--;
	}
	return min;
}



Mat rotateImg(Mat img, double core);
/****************进行霍夫直线变换,画出*******************/
Mat findLines(Mat img)
{
	Mat dst_output, dst, result_map;
	float cols = img.cols;
	float rows = img.rows;
	float resultLong = (100.0 / 440.0)*cols;//车牌标准长度
	float resultRows = (90.0 / 140.0)*rows;//车牌上下标准间距
	float core;
	cvtColor(img, dst_output, CV_BGR2GRAY);

	Mat contours;
	Canny(dst_output, contours, 100, 300);
	cvtColor(contours, dst, CV_GRAY2BGR);
	threshold(contours, contours, 115, 255, THRESH_BINARY);
	vector<Vec4f>lines;

	HoughLinesP(contours, lines, 1, CV_PI / 180, 130, resultLong, resultRows);//110
	Scalar color = Scalar(0, 0, 255);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cout << endl << "有一条";
		Vec4f pline = lines[i];
		line(dst, Point(pline[0], pline[1]), Point(pline[2], pline[3]), color, 3, LINE_AA);
		core = abs(pline[1] - pline[3]) / abs(pline[0] - pline[2]);
		cout << "(" << pline[0] << "," << pline[1] << ")" << "(" << pline[2] << "," << pline[3] << ")";
	}
	core = 180 / CV_PI*atan(core);;
	cout << endl << "角度是" << core;

	result_map = rotateImg(img, core);

	imshow("霍夫直线", dst);
	imshow("霍夫", result_map);

	return result_map;
}
/******************************************/








/*************角点提取*******************/
void CoreGet(Mat img)
{
	Mat dstImage;//目标图  
	Mat normImage;//归一化后的图  
	Mat scaledImage;//线性变换后的八位无符号整型的图  
	int thresh = 30;
	cvtColor(img, img, CV_BGR2GRAY);
	cornerHarris(img,  //Input single-channel 8-bit or floating-point image.  
		dstImage,  //Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .  
		2,         //Neighborhood size  
		3,         //Aperture parameter for the Sobel() operator  
		0.04,      // Harris detector free parameter  
		BORDER_DEFAULT);// Pixel extrapolation method  
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//将归一化后的图线性变换成8位无符号整型   
	for (int j = 0; j < normImage.rows; j++){
		for (int i = 0; i < normImage.cols; i++){
			if ((int)normImage.at<float>(j, i) > thresh + 100){
				circle(img, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	imshow("角点", img);
}


/****************************************/



Mat findBlue(Mat img)
{
	Mat dst;

	Mat dst_hsv, res; //分割后图像  
	int cols = img.cols;
	int rows = img.rows;

	cvtColor(img, dst_hsv, CV_BGR2HSV);
	Mat imgH;
	Mat imgS;
	Mat imgV;
	vector<Mat> channels;
	split(dst_hsv, channels);

	imgH = channels.at(0);
	imgS = channels.at(1);
	imgV = channels.at(2);

	vector<Point> blue;

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			//////////////////h通道/-/////////////////////
			uchar hvalue = imgH.at<uchar>(i, j);  //
			//////////////////s通道/-/////////////////////
			uchar svalue = imgS.at<uchar>(i, j);  //
			//////////////////v通道/-/////////////////////
			uchar vvalue = imgV.at<uchar>(i, j);  //
			//  //if ((hvalue>90 && hvalue<120) && (svalue>80 && svalue<220) && (vvalue>80 && vvalue < 255))

			//if ((hvalue>90 && hvalue<120) && (svalue>95 && svalue<235) && (vvalue>80 && vvalue < 255))
			if ((hvalue>90 && hvalue<120) && (svalue>95 && svalue<260) && (vvalue>80 && vvalue < 255))
			{
				blue.push_back(Point(i, j));
				continue;
			}
			else
			{
				imgH.at<uchar>(i, j) = 0;
				imgS.at<uchar>(i, j) = 0;
				imgV.at<uchar>(i, j) = 0;
			}


		}

	}
	channels.at(0) = imgH;
	channels.at(1) = imgS;
	channels.at(0) = imgV;
	res = Mat(img.size(), CV_8UC3);
	merge(channels, res);
	//cvtColor(res, res, CV_HSV2BGR);

	int maxX = findPointXMax(blue);
	int maxY = findPointYMax(blue);
	int minX = findPointXMin(blue);
	int minY = findPointYMin(blue);

	cout << endl << minX << " " << minY;

	res = findLines(res);//霍夫直线变换矫正图片
	//CoreGet(res);//角点检测

	imshow("进一步处理", res);
	//cout << endl<<res.type();
	//Mat newWin = Mat::zeros(maxX - minX, maxY - minY, res.type());

	Rect rect(minY, minX, maxY - minY, maxX - minX);
	Mat per = res(rect);

	//findLines(per);//直线查找
	//CoreGet(per);//角点检测
	return per;
}

Mat deleteEndWhite(Mat img)//去除白边
{
	//Mat dst;
	int cols = img.cols;
	int rows = img.rows;
	uchar change;
	for (size_t i = 0; i < rows; i++)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t j = 0; j < cols; j++)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) < 255)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 0;
			}
		}
	}


	for (size_t i = rows - 1; i >0; i--)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t j = cols - 1; j > 0; j--)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) < 255)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 0;
			}
		}
	}


	return img;
}


Mat deleteEndBlack(Mat img)//去除黑边
{
	//Mat dst;
	int cols = img.cols;
	int rows = img.rows;
	uchar change;
	//int a = img.at<uchar>(3, 102);
	//cout <<endl<<"这个不是黑色?" <<a;

	for (size_t i = 0; i < rows; i++)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t j = 0; j < cols; j++)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) >0)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 255;
			}
		}
	}




	for (size_t j = 0; j < cols; j++)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t i = 0; i < rows; i++)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) >0)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 255;
			}
		}
	}


	for (size_t j = cols - 1; j >0; j--)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t i = rows - 1; i >0; i--)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) >0)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 255;
			}
		}
	}

	for (size_t i = rows - 1; i >0; i--)
	{
		//uchar *p = img.ptr<uchar>(i);
		for (size_t j = cols - 1; j > 0; j--)
		{
			//cout << img.at<uchar>(i, j);
			if (img.at<uchar>(i, j) >0)
			{
				break;
			}
			else{
				img.at<uchar>(i, j) = 255;
			}
		}
	}
	return img;
}


double findCore(Mat img)//前提是灰度图，计算角度
{
	Mat dst;
	double tanCore;
	int rows = img.rows;
	int cols = img.cols;
	double rowsLine = 0, colsLine = 0;
	for (size_t i = 0; i < rows; i++)
	{
		if (img.at<uchar>(i, 0) == 255) rowsLine++;
		else if (img.at<uchar>(i, 0) != 255)break;
	}
	for (size_t j = 0; j < cols; j++)
	{
		if (img.at<uchar>(0, j) == 255) colsLine++;
		else if (img.at<uchar>(0, j) != 255)break;
	}

	if (rowsLine == 0 || colsLine == 0) tanCore = 0;
	else tanCore = rowsLine / colsLine;

	//double core =180/CV_PI*atan(rowsLine / colsLine);
	double core = 180 / CV_PI*atan(tanCore);



	return core;
}


Mat rotateImg(Mat img, double core)
{
	Mat dst;
	//cout << endl << rowsLine / colsLine;
	//旋转角度
	int centreX = img.rows / 2;
	int centreY = img.cols / 2;
	//	Mat rot_mat(2, 3, CV_32FC1);
	Mat rot_mat = getRotationMatrix2D(Point(centreX, centreY), -core, 1);
	warpAffine(img, dst, rot_mat, img.size());

	return dst;
}


//车牌大小为440×140；
vector<Mat>vecb;//分割后的图片集
void getSomeChar(Mat img)
{
	Mat dst;
	int rows = img.rows;
	int cols = img.cols;
	bool blackPoint = false;
	bool whitePoint = false;
	int whiteDistance = 0;//记录一个字母的区间大小
	int BlackPointY, WhitePointY;//记录黑色或白色区域第一个点Y的值
	int m = 0;//记录字符个数
	double carPostResult = 45.0 / 440.0;//车牌字体标准大小
	double carPointResult = 10.0 / 440.0;//车牌0的标准大小
	cout << endl << "cols=" << cols << "标准" << carPostResult*cols;
	size_t i = 0, j = 0;
	for (j = 0; j < cols; j++)
	{
		for (i = 0; i < rows; i++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				if (!whitePoint)
				{
					WhitePointY = j;
					whitePoint = true;
					blackPoint = false;
				}
				//whiteDistance++;
				//cout << whiteDistance<<"  ";
				break;
			}
			//else{
			//	//cout << whiteDistance;
			//	if (deleteEndBlack==false && whiteDistance > 20)//比如U的顶点的距离中间有黑色的部分，一个字母或者数字一般为30以下
			//	{
			//		BlackPointY = j;
			//		blackPoint = true;
			//		whitePoint = false;
			//		//Mat dst = Mat(rows, BlackPointY - WhitePointY, img.type());

			//		Mat dst;//截取
			//		Rect rect(WhitePointY, 0, BlackPointY - WhitePointY, rows);
			//		dst = img(rect);
			//		imshow("wrong" + m, dst);
			//		vecb.push_back(dst);


			//		m++;
			//		whiteDistance = 0;//距离清零
			//	}

			//}
		}
		if (i == rows&&whitePoint == true)
		{
			whitePoint = false;
			blackPoint = true;

			Mat dst;//截取
			float car;
			if (j - WhitePointY <= carPointResult*cols&&vecb.size() == 2)//判断是否为一点，方框就变小；
			{
				car = j - WhitePointY;
			}
			else if (carPostResult*cols > j - WhitePointY)//标准字体大小，防止有偏旁的字被分割
			{
				car = carPostResult*cols;
				j = WhitePointY + carPostResult*cols;
			}
			else
			{
				car = j - WhitePointY;
			}
			Rect rect(WhitePointY, 0, car, rows);

			dst = img(rect);
			//imshow("wrong" + m, dst);
			vecb.push_back(dst);
		}
	}
}




int main()
{
	Mat img, imgGauss, imgGray, imgSobel, imgThreshold, imgMorph, imgCont, dst;
	img = imread("E://test/car2.jpg");
	namedWindow(Input_Title, CV_WINDOW_AUTOSIZE);
	imshow(Input_Title, img);

	cvtColor(img, imgGray, CV_BGR2GRAY);

	Mat img_blur;
	medianBlur(imgGray, img_blur, 3);

	Mat img_canny;
	Canny(imgGray, img_canny, 500, 200, 3);

	Mat img_good;//膨胀腐蚀操作
	img_good = erode_dilate_Pict(img_canny);
	imshow("good", img_good);


	Mat img_temp = findCar(img_good, img);
	imshow("pict", img_temp);



	Mat img_deleteBlack;//把蓝色去除掉
	img_deleteBlack = findBlue(img_temp);
	imshow("出去蓝色以为其他颜色", img_deleteBlack);

	Mat img_dilateAgain;//闭操作去除多余黑点
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//原Size(5,5)
	//Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	//dilate(img_deleteBlack, img_dilateAgain, element);
	morphologyEx(img_deleteBlack, img_dilateAgain, MORPH_CLOSE, element);
	imshow("show", ~img_dilateAgain);

	Mat img_grayAgin;//再次灰度
	//img_dilateAgain = ~img_dilateAgain;//原代码
	img_dilateAgain = ~img_deleteBlack;//试试

	cvtColor(img_dilateAgain, img_grayAgin, CV_BGR2GRAY);

	morphologyEx(img_grayAgin, img_grayAgin, MORPH_OPEN, element);//试试

	imshow("再次灰度", img_grayAgin);




	/****我要再试试，要删除哦Begin****/
	Mat Atry;
	threshold(img_grayAgin, Atry, 200, 255, THRESH_BINARY);
	imshow("这里二值化试试", ~Atry);
	Atry = ~Atry;
	size_t rows = Atry.rows;
	size_t cols = Atry.cols;
	size_t i, j;
	Mat eee = getStructuringElement(MORPH_RECT, Size(3, 1));
	Mat aaa = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(Atry, Atry, eee, Point(-1, -1), 1);
	morphologyEx(Atry, Atry, MORPH_CLOSE, aaa);
	imshow("试一试这个膨胀", Atry);
	Mat AAtry = deleteEndBlack(Atry);
	imshow("hhhh", ~Atry);
	Atry = ~Atry;

	/****我要再试试，要删除哦End****/


	//double core = findCore(img_grayAgin);//原版
	double core = findCore(Atry);//找到旋转角度
	//cout <<endl <<"角度是" << core;
	//Mat endBlack = deleteEndWhite(img_grayAgin);//原版
	Mat endBlack = deleteEndWhite(Atry);//去白边
	imshow("去百变", endBlack);
	endBlack = rotateImg(endBlack, core);//旋转
	imshow("倒数第二个步骤", endBlack);

	//二值化,变为黑白图
	//threshold(img_grayAgin, endBlack, 115, 255, THRESH_BINARY);//100
	threshold(endBlack, img_grayAgin, 115, 255, THRESH_BINARY);//100
	imshow("阈值化", img_grayAgin);

	//腐蚀，使字体变细,便于识别
	Mat try1try;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//dilate(img_grayAgin, img_grayAgin, element2);
	erode(img_grayAgin, try1try, element1, Point(-1, -1));

	//morphologyEx(try1try, try1try, MORPH_OPEN, element,Point(-1,-1));
	//erode(try1try, try1try, element2, Point(-1, -1));
	//dilate(try1try, try1try, element1, Point(-1, -1));
	imshow("试一试", try1try);

	//分割图片
	getSomeChar(try1try);
	//char addressURL;
	char *addressURL = "E://test/car/car1/";
	char url[100];
	vector<Mat>::iterator it = vecb.begin();
	int a = 0;
	while (it != vecb.end())
	{

		sprintf(url, "E://test/car/car6/%d.jpg", a);
		//addressURL += aa;
		cout << endl << "adressURL" << url;
		//imwrite(url, *it);
		imshow(url, *it);
		it++;
		a++;
	}

	waitKey(0);
	destroyAllWindows();

	return 0;
}