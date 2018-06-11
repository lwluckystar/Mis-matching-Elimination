// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "Mis_matcherE.h"

void GmsMatch(Mat &img1, Mat &img2);


//����ͼ�񣬶�ͼƬ��С��������
void runImagePair(){

	Mat img1 = imread("P1010517.JPG");
	Mat img2 = imread("P1010520.JPG");
	//����imresize��������ͼƬ��С���иı�
	imresize(img1, 680);
	imresize(img2, 680);
	//����GmsMatch��������ͼƬ�������������������ƥ��
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
	double t3 = (double)getTickCount();  //״̬���¼ӡ��洢�㷨�ܵ�����ʱ��
	double t1 = getTickCount();//״̬���¼ӡ��洢orb�㷨�ļ���ʱ��

	vector<KeyPoint> kp1, kp2;//�洢��⵽�������㣬���ݽṹΪKeyPoint
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;//DMatch��Ҫ��������ƥ����Ϣ�Ľṹ��

	Ptr<ORB> orb = ORB::create(3000);//����ORB���ͼƬ�е������㣬�������������Ϊ1000

    //Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce"); //�¼ӣ���������ƥ����

	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);
	t1 = ((double)getTickCount() - t1) / getTickFrequency();//״̬���¼ӡ�orb�㷨����ʱ�乫ʽ
	cout << "ORB�㷨��ʱ��" << t1 << "��" << endl;//״̬���¼�
	cout << "ͼ��1���������:" << kp1.size() << endl;//״̬���¼�
	cout << "ͼ��2���������:" << kp2.size() << endl;//״̬���¼�
	cout << "ͼ��1�������������С��" << d1.size()//״̬���¼�
		<< "����������������" << d1.rows << "��ά����" << d1.cols << endl;//״̬���¼�
	cout << "ͼ��2�������������С��" << d2.size()//״̬���¼�
		<< "����������������" << d2.rows << "��ά����" << d2.cols << endl;//״̬���¼�
	
	//Mat  descriptors1,descriptors2;//�¼ӣ�
	//descriptor_extractor->compute( img1, kp1, descriptors1 ); //�¼ӣ�     
	//descriptor_extractor->compute( img2, kp2, descriptors2 ); //�¼ӣ�   



#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	Mat imgMatch;//״̬���¼ӡ�
	imgMatch= DrawInlier(img1, img2, kp1, kp2, matches_all, 1);//״̬���¼ӡ�
	imwrite("��ƥ��10000.jpg", imgMatch);
	namedWindow("imgMatch", CV_WINDOW_FREERATIO);//�¼ӣ�
	imshow("imgMatch", imgMatch);//�¼ӣ�
	waitKey(10);
#endif

	// GMS filter ��GMS��������׼���
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	//����
	double t2 = getTickCount();//״̬���¼ӡ��洢GMS�޳���ƥ��ʱ��
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();//״̬���¼ӡ�GMS�޳���ƥ��ʱ��
	cout << "GMS�޳���ƥ��ʱ�䣺" << t2 << "��" << endl;//״̬���¼�
	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches��������׼��֮�����
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
	cout << "goodMatch������" << matches_gms.size() << endl;//״̬���¼ӡ�ʣ��������ƥ���

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	imwrite("Result_1_10000.jpg", show);
	namedWindow("show", CV_WINDOW_FREERATIO);
	imshow("show", show);
	t3 = double((getTickCount() - t3) / getTickFrequency()); //״̬���洢�㷨�ܵ�����ʱ��
	cout << "�㷨�ܵ�����ʱ�䣺" << t3 << "��" << endl;//״̬���¼�
	waitKey();
}


