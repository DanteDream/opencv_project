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
	Sobel(img, img, img.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);//x����
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
	vector<vector<Point>> contourPoints;//һ������
	vector<Vec4i> hierachy;


}


Mat erode_dilate_Pict(Mat img)
{
	Mat img_good, img_erode, img_dilate;
	Mat elementX = getStructuringElement(MORPH_RECT, Size(25, 1));
	Mat elementY = getStructuringElement(MORPH_RECT, Size(1, 19));//(1,19)
	Point point(-1, -1);
	//x�������͸�ʴ
	dilate(img, img_dilate, elementX, point, 3);//3
	erode(img_dilate, img_erode, elementX, point, 5);//5
	dilate(img_erode, img_dilate, elementX, point, 3);//3
	//y�������͸�ʴ
	//dilate(img_dilate, img_dilate, elementY, point, 1);//û�е�
	erode(img_dilate, img_erode, elementY, point, 2);//2
	dilate(img_erode, img_dilate, elementY, point, 3);//3

	//������������ֵ�˲�
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
/****************���л���ֱ�߱任,����*******************/
Mat findLines(Mat img)
{
	Mat dst_output, dst, result_map;
	float cols = img.cols;
	float rows = img.rows;
	float resultLong = (100.0 / 440.0)*cols;//���Ʊ�׼����
	float resultRows = (90.0 / 140.0)*rows;//�������±�׼���
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
		cout << endl << "��һ��";
		Vec4f pline = lines[i];
		line(dst, Point(pline[0], pline[1]), Point(pline[2], pline[3]), color, 3, LINE_AA);
		core = abs(pline[1] - pline[3]) / abs(pline[0] - pline[2]);
		cout << "(" << pline[0] << "," << pline[1] << ")" << "(" << pline[2] << "," << pline[3] << ")";
	}
	core = 180 / CV_PI*atan(core);;
	cout << endl << "�Ƕ���" << core;

	result_map = rotateImg(img, core);

	imshow("����ֱ��", dst);
	imshow("����", result_map);

	return result_map;
}
/******************************************/








/*************�ǵ���ȡ*******************/
void CoreGet(Mat img)
{
	Mat dstImage;//Ŀ��ͼ  
	Mat normImage;//��һ�����ͼ  
	Mat scaledImage;//���Ա任��İ�λ�޷������͵�ͼ  
	int thresh = 30;
	cvtColor(img, img, CV_BGR2GRAY);
	cornerHarris(img,  //Input single-channel 8-bit or floating-point image.  
		dstImage,  //Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .  
		2,         //Neighborhood size  
		3,         //Aperture parameter for the Sobel() operator  
		0.04,      // Harris detector free parameter  
		BORDER_DEFAULT);// Pixel extrapolation method  
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//����һ�����ͼ���Ա任��8λ�޷�������   
	for (int j = 0; j < normImage.rows; j++){
		for (int i = 0; i < normImage.cols; i++){
			if ((int)normImage.at<float>(j, i) > thresh + 100){
				circle(img, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	imshow("�ǵ�", img);
}


/****************************************/



Mat findBlue(Mat img)
{
	Mat dst;

	Mat dst_hsv, res; //�ָ��ͼ��  
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
			//////////////////hͨ��/-/////////////////////
			uchar hvalue = imgH.at<uchar>(i, j);  //
			//////////////////sͨ��/-/////////////////////
			uchar svalue = imgS.at<uchar>(i, j);  //
			//////////////////vͨ��/-/////////////////////
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

	res = findLines(res);//����ֱ�߱任����ͼƬ
	//CoreGet(res);//�ǵ���

	imshow("��һ������", res);
	//cout << endl<<res.type();
	//Mat newWin = Mat::zeros(maxX - minX, maxY - minY, res.type());

	Rect rect(minY, minX, maxY - minY, maxX - minX);
	Mat per = res(rect);

	//findLines(per);//ֱ�߲���
	//CoreGet(per);//�ǵ���
	return per;
}

Mat deleteEndWhite(Mat img)//ȥ���ױ�
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


Mat deleteEndBlack(Mat img)//ȥ���ڱ�
{
	//Mat dst;
	int cols = img.cols;
	int rows = img.rows;
	uchar change;
	//int a = img.at<uchar>(3, 102);
	//cout <<endl<<"������Ǻ�ɫ?" <<a;

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


double findCore(Mat img)//ǰ���ǻҶ�ͼ������Ƕ�
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
	//��ת�Ƕ�
	int centreX = img.rows / 2;
	int centreY = img.cols / 2;
	//	Mat rot_mat(2, 3, CV_32FC1);
	Mat rot_mat = getRotationMatrix2D(Point(centreX, centreY), -core, 1);
	warpAffine(img, dst, rot_mat, img.size());

	return dst;
}


//���ƴ�СΪ440��140��
vector<Mat>vecb;//�ָ���ͼƬ��
void getSomeChar(Mat img)
{
	Mat dst;
	int rows = img.rows;
	int cols = img.cols;
	bool blackPoint = false;
	bool whitePoint = false;
	int whiteDistance = 0;//��¼һ����ĸ�������С
	int BlackPointY, WhitePointY;//��¼��ɫ���ɫ�����һ����Y��ֵ
	int m = 0;//��¼�ַ�����
	double carPostResult = 45.0 / 440.0;//���������׼��С
	double carPointResult = 10.0 / 440.0;//����0�ı�׼��С
	cout << endl << "cols=" << cols << "��׼" << carPostResult*cols;
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
			//	if (deleteEndBlack==false && whiteDistance > 20)//����U�Ķ���ľ����м��к�ɫ�Ĳ��֣�һ����ĸ��������һ��Ϊ30����
			//	{
			//		BlackPointY = j;
			//		blackPoint = true;
			//		whitePoint = false;
			//		//Mat dst = Mat(rows, BlackPointY - WhitePointY, img.type());

			//		Mat dst;//��ȡ
			//		Rect rect(WhitePointY, 0, BlackPointY - WhitePointY, rows);
			//		dst = img(rect);
			//		imshow("wrong" + m, dst);
			//		vecb.push_back(dst);


			//		m++;
			//		whiteDistance = 0;//��������
			//	}

			//}
		}
		if (i == rows&&whitePoint == true)
		{
			whitePoint = false;
			blackPoint = true;

			Mat dst;//��ȡ
			float car;
			if (j - WhitePointY <= carPointResult*cols&&vecb.size() == 2)//�ж��Ƿ�Ϊһ�㣬����ͱ�С��
			{
				car = j - WhitePointY;
			}
			else if (carPostResult*cols > j - WhitePointY)//��׼�����С����ֹ��ƫ�Ե��ֱ��ָ�
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

	Mat img_good;//���͸�ʴ����
	img_good = erode_dilate_Pict(img_canny);
	imshow("good", img_good);


	Mat img_temp = findCar(img_good, img);
	imshow("pict", img_temp);



	Mat img_deleteBlack;//����ɫȥ����
	img_deleteBlack = findBlue(img_temp);
	imshow("��ȥ��ɫ��Ϊ������ɫ", img_deleteBlack);

	Mat img_dilateAgain;//�ղ���ȥ������ڵ�
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//ԭSize(5,5)
	//Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	//dilate(img_deleteBlack, img_dilateAgain, element);
	morphologyEx(img_deleteBlack, img_dilateAgain, MORPH_CLOSE, element);
	imshow("show", ~img_dilateAgain);

	Mat img_grayAgin;//�ٴλҶ�
	//img_dilateAgain = ~img_dilateAgain;//ԭ����
	img_dilateAgain = ~img_deleteBlack;//����

	cvtColor(img_dilateAgain, img_grayAgin, CV_BGR2GRAY);

	morphologyEx(img_grayAgin, img_grayAgin, MORPH_OPEN, element);//����

	imshow("�ٴλҶ�", img_grayAgin);




	/****��Ҫ�����ԣ�Ҫɾ��ŶBegin****/
	Mat Atry;
	threshold(img_grayAgin, Atry, 200, 255, THRESH_BINARY);
	imshow("�����ֵ������", ~Atry);
	Atry = ~Atry;
	size_t rows = Atry.rows;
	size_t cols = Atry.cols;
	size_t i, j;
	Mat eee = getStructuringElement(MORPH_RECT, Size(3, 1));
	Mat aaa = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(Atry, Atry, eee, Point(-1, -1), 1);
	morphologyEx(Atry, Atry, MORPH_CLOSE, aaa);
	imshow("��һ���������", Atry);
	Mat AAtry = deleteEndBlack(Atry);
	imshow("hhhh", ~Atry);
	Atry = ~Atry;

	/****��Ҫ�����ԣ�Ҫɾ��ŶEnd****/


	//double core = findCore(img_grayAgin);//ԭ��
	double core = findCore(Atry);//�ҵ���ת�Ƕ�
	//cout <<endl <<"�Ƕ���" << core;
	//Mat endBlack = deleteEndWhite(img_grayAgin);//ԭ��
	Mat endBlack = deleteEndWhite(Atry);//ȥ�ױ�
	imshow("ȥ�ٱ�", endBlack);
	endBlack = rotateImg(endBlack, core);//��ת
	imshow("�����ڶ�������", endBlack);

	//��ֵ��,��Ϊ�ڰ�ͼ
	//threshold(img_grayAgin, endBlack, 115, 255, THRESH_BINARY);//100
	threshold(endBlack, img_grayAgin, 115, 255, THRESH_BINARY);//100
	imshow("��ֵ��", img_grayAgin);

	//��ʴ��ʹ�����ϸ,����ʶ��
	Mat try1try;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//dilate(img_grayAgin, img_grayAgin, element2);
	erode(img_grayAgin, try1try, element1, Point(-1, -1));

	//morphologyEx(try1try, try1try, MORPH_OPEN, element,Point(-1,-1));
	//erode(try1try, try1try, element2, Point(-1, -1));
	//dilate(try1try, try1try, element1, Point(-1, -1));
	imshow("��һ��", try1try);

	//�ָ�ͼƬ
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