// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "Mis_matcherE.h"

void GmsMatch(Mat &img1, Mat &img2);


//读入图像，对图片大小进行缩放
void runImagePair(){

	Mat img1 = imread("P1010517.JPG");
	Mat img2 = imread("P1010520.JPG");
	//调用imresize函数，对图片大小进行改变
	imresize(img1, 680);
	imresize(img2, 680);
	//调用GmsMatch函数，对图片进行特征点检测和特征点匹配
	GmsMatch(img1, img2);
}


int main()
{

#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU
	runImagePair();
	return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
	double t3 = (double)getTickCount();  //状态：新加。存储算法总的运行时间
	double t1 = getTickCount();//状态：新加。存储orb算法的计算时间

	vector<KeyPoint> kp1, kp2;//存储检测到的特征点，数据结构为KeyPoint
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;//DMatch主要用来储存匹配信息的结构体

	Ptr<ORB> orb = ORB::create(3000);//调用ORB检测图片中的特征点，检测特征点数量为1000

    //Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce"); //新加，创建特征匹配器

	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);
	t1 = ((double)getTickCount() - t1) / getTickFrequency();//状态：新加。orb算法计算时间公式
	cout << "ORB算法用时：" << t1 << "秒" << endl;//状态：新加
	cout << "图像1特征点个数:" << kp1.size() << endl;//状态：新加
	cout << "图像2特征点个数:" << kp2.size() << endl;//状态：新加
	cout << "图像1特征描述矩阵大小：" << d1.size()//状态：新加
		<< "，特征向量个数：" << d1.rows << "，维数：" << d1.cols << endl;//状态：新加
	cout << "图像2特征描述矩阵大小：" << d2.size()//状态：新加
		<< "，特征向量个数：" << d2.rows << "，维数：" << d2.cols << endl;//状态：新加
	
	//Mat  descriptors1,descriptors2;//新加，
	//descriptor_extractor->compute( img1, kp1, descriptors1 ); //新加，     
	//descriptor_extractor->compute( img2, kp2, descriptors2 ); //新加，   



#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	Mat imgMatch;//状态：新加。
	imgMatch= DrawInlier(img1, img2, kp1, kp2, matches_all, 1);//状态：新加。
	imwrite("粗匹配10000.jpg", imgMatch);
	namedWindow("imgMatch", CV_WINDOW_FREERATIO);//新加，
	imshow("imgMatch", imgMatch);//新加，
	waitKey(10);
#endif

	// GMS filter ，GMS过滤误配准点对
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	//调用
	double t2 = getTickCount();//状态：新加。存储GMS剔除误匹配时间
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();//状态：新加。GMS剔除误匹配时间
	cout << "GMS剔除误匹配时间：" << t2 << "秒" << endl;//状态：新加
	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches，画出配准点之间的线
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
	cout << "goodMatch个数：" << matches_gms.size() << endl;//状态：新加。剩余的优秀的匹配点

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	imwrite("Result_1_10000.jpg", show);
	namedWindow("show", CV_WINDOW_FREERATIO);
	imshow("show", show);
	t3 = double((getTickCount() - t3) / getTickFrequency()); //状态：存储算法总的运行时间
	cout << "算法总的运行时间：" << t3 << "秒" << endl;//状态：新加
	waitKey();
}


