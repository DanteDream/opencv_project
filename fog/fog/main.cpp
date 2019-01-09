#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
using namespace std;
using namespace cv;

Mat whatChannelsIsMin;
vector<Mat> channels(3);
Mat fImage;
vector<double> A(3);//分别存取A_b,A_g,A_r  
//2.2在求第三步的t(x)时，会用到以下的矩阵，这里可以提前求出  用在第二步，第三步需要
vector<Mat> fImageBorderVectorA(3);



Mat splitBGR(Mat src,int hPatch,int vPatch)//寻找BGR中最大的值
{


	int width = src.cols;
	int height = src.rows;
	
	src.convertTo(fImage, CV_32FC3, 1.0 / 255, 0);
	copyMakeBorder(fImage, fImage, vPatch / 2, vPatch / 2, hPatch / 2, hPatch / 2, BORDER_REPLICATE);//添加边框


	Mat src_dark(height,width, CV_32FC1);

	
	
	split(fImage, channels);

	Mat B_image(height,width, CV_8UC1), G_image(height, width, CV_8UC1), R_image(height, width, CV_8UC1);


	double minTemp, minPixel;

	for (int i = 0; i < height; i++)
	{

		for (int j = 0; j < width; j++)
		{

			minPixel = 1.0;
			for (vector<Mat>::iterator it = channels.begin(); it != channels.end(); it++)
			{
				Mat roi(*it, Rect(j, i, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			src_dark.at<float>(i, j) = float(minPixel);

		}
	}

	imshow("哈哈", src_dark);

	return src_dark;

}

//第二步
void a_second(Mat src, int hPatch, int vPatch)
{
	/*第2步：求出 A(global atmospheric light)*/
//2.1 计算出darkChannel中,前top个亮的值,论文中取值为0.1%  
	float top = 0.001;
	float numberTop = top * src.rows*src.cols;
	Mat darkChannelVector;
	darkChannelVector = src.reshape(1, 1);
	Mat_<int> darkChannelVectorIndex;
	sortIdx(darkChannelVector, darkChannelVectorIndex, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);//排序
	//制作掩码  
	Mat mask(darkChannelVectorIndex.rows, darkChannelVectorIndex.cols, CV_8UC1);//注意mask的类型必须是CV_8UC1  
	for (unsigned int r = 0; r < darkChannelVectorIndex.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannelVectorIndex.cols; c++)
		{
			if (darkChannelVectorIndex.at<int>(r, c) <= numberTop)
				mask.at<uchar>(r, c) = 1;
			else
				mask.at<uchar>(r, c) = 0;
		}
	}
	Mat darkChannelIndex = mask.reshape(1, src.rows);


	vector<double>::iterator itA = A.begin();
	vector<Mat>::iterator it = channels.begin();	
	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();

	for (; it != channels.end() && itA != A.end() && itAA != fImageBorderVectorA.end(); it++, itA++, itAA++)
	{
		Mat roi(*it, Rect(hPatch / 2, vPatch / 2, src.cols, src.rows));
		minMaxLoc(roi, 0, &(*itA), 0, 0, darkChannelIndex);//  
		(*itAA) = (*it) / (*itA); //[注意：这个地方有除号，但是没有判断是否等于0]  
	}
}


//第三步
Mat t_thrid(Mat src,int hPatch,int vPatch)
{
	/*第三步：求t(x)*/
	double minTemp, minPixel;
	Mat darkChannelA(src.rows, src.cols, CV_32FC1);
	float omega = 0.95;//0<w<=1,论文中取值为0.95  
	//代码和求darkChannel的时候,代码差不多  
	vector<Mat>::iterator itAA;
	for (unsigned int r = 0; r < src.rows; r++)
	{
		for (unsigned int c = 0; c < src.cols; c++)
		{
			minPixel = 1.0;
			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++)
			{
				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannelA.at<float>(r, c) = float(minPixel);
		}
	}
	Mat tx = 1.0 - omega * darkChannelA;
	return tx;
}

Mat result_(Mat src,Mat tx)
{
	/*第四步：我们可以求J(x)*/
	float t0 = 0.1;//论文中取t0 = 0.1  
	Mat jx(src.rows, src.cols, CV_32FC3);
	for (size_t r = 0; r < jx.rows; r++)
	{
		for (size_t c = 0; c < jx.cols; c++)
		{
			jx.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);
		}
	}
	namedWindow("jx", 1);
	imshow("jx", jx);
	Mat jx8U;
	jx.convertTo(jx8U, CV_8UC3, 255, 0);
	return jx;
	//imwrite("jx.jpg", jx8U);
}

int main()
{
	Mat src;
	Mat distance;
	int hPatch = 21;
	int vPatch = 21;
	src = imread("G:\\ImageTest\\fog2.jpg");


	//GaussianBlur(src,src,Size(3,3),0,0);

	imshow("原图", src);
	Mat src_dark=splitBGR(src, hPatch, hPatch);//获取暗通道

	a_second(src_dark, hPatch, hPatch);
	Mat t=t_thrid(src_dark, hPatch, hPatch);
	Mat result=result_(src, t);
	
	

	waitKey(0);
	destroyAllWindows();
}








//void result_A(Mat src, Mat image,int hPatch, int vPatch)
//{
//	/*第2步：求出 A(global atmospheric light)*/
////2.1 计算出darkChannel中,前top个亮的值,论文中取值为0.1%  
//	float top = 0.001;
//	float numberTop = top * src.rows*src.cols;
//	Mat darkChannelVector;
//	darkChannelVector = src.reshape(1, 1);
//	Mat_<int> darkChannelVectorIndex;
//	sortIdx(darkChannelVector, darkChannelVectorIndex, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);//排序
//	//制作掩码  
//	Mat mask(darkChannelVectorIndex.rows, darkChannelVectorIndex.cols, CV_8UC1);//注意mask的类型必须是CV_8UC1  
//	for (unsigned int r = 0; r < darkChannelVectorIndex.rows; r++)
//	{
//		for (unsigned int c = 0; c < darkChannelVectorIndex.cols; c++)
//		{
//			if (darkChannelVectorIndex.at<int>(r, c) <= numberTop)
//				mask.at<uchar>(r, c) = 1;
//			else
//				mask.at<uchar>(r, c) = 0;
//		}
//	}
//	Mat darkChannelIndex = mask.reshape(1, src.rows);
//
//	vector<double> A(3);//分别存取A_b,A_g,A_r  
//	vector<double>::iterator itA = A.begin();
//	vector<Mat>::iterator it = channels.begin();
//	//2.2在求第三步的t(x)时，会用到以下的矩阵，这里可以提前求出  用在第二步，第三步需要
//	vector<Mat> fImageBorderVectorA(3);
//	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
//	
//	for (; it != channels.end() && itA != A.end() && itAA != fImageBorderVectorA.end(); it++, itA++, itAA++)
//	{
//		Mat roi(*it, Rect(hPatch / 2, vPatch / 2, src.cols, src.rows));
//		minMaxLoc(roi, 0, &(*itA), 0, 0, darkChannelIndex);//  
//		(*itAA) = (*it) / (*itA); //[注意：这个地方有除号，但是没有判断是否等于0]  
//	}
//
//
//	/*第三步：求t(x)*/
//	double minTemp, minPixel;
//	Mat darkChannelA(src.rows, src.cols, CV_32FC1);
//	float omega = 0.95;//0<w<=1,论文中取值为0.95  
//	//代码和求darkChannel的时候,代码差不多  
//	for (unsigned int r = 0; r < src.rows; r++)
//	{
//		for (unsigned int c = 0; c < src.cols; c++)
//		{
//			minPixel = 1.0;
//			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++)
//			{
//				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
//				minMaxLoc(roi, &minTemp);
//				minPixel = min(minPixel, minTemp);
//			}
//			darkChannelA.at<float>(r, c) = float(minPixel);
//		}
//	}
//	Mat tx = 1.0 - omega * darkChannelA;
//
//
//	/*第四步：我们可以求J(x)*/
//	float t0 = 0.1;//论文中取t0 = 0.1  
//	Mat jx(image.rows, image.cols, CV_32FC3);
//	for (size_t r = 0; r < jx.rows; r++)
//	{
//		for (size_t c = 0; c < jx.cols; c++)
//		{
//			jx.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);
//		}
//	}
//	namedWindow("jx", 1);
//	imshow("jx", jx);
//	Mat jx8U;
//	jx.convertTo(jx8U, CV_8UC3, 255, 0);
//	imwrite("jx.jpg", jx8U);
//}