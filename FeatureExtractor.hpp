/*  
 *  FeatureExtractor.hpp
 *  FeatureExtractor
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_FEATUREEXTRACTOR_HPP
#define DEFINE_FEATUREEXTRACTOR_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "commonUse.hpp"
#include "DataPreparation.hpp"
using namespace std;
using namespace cv;

struct FeatureInfo{
	int seqNum;
	int frameNum;
	Rect box;
	Mat hog;
	Mat shape;
	Mat edge;
	Mat dt;
};

class FeatureExtractor
{
public:
	vector<string> _featureType;
	double _thresContour;
	int _version;
	vector<vector<FeatureInfo> > _featureInfo;

	FeatureExtractor() {}
	int initialize(ConfigFile &cfg);
	int compute(string workDir);
	int getFeatureFromIntel(vector<vector<HandInfo> > &handInfo);
	int getFeatureAllFromIntel(vector<vector<HandInfo> > &handInfo);
	int getFeatureAllFromGTEA(vector<vector<HandInfo> > &handInfo);

	int getFeature_SIFT_BOW(Mat &img, Mat &clusterCenters, Mat &output);
	int getFeature_HOG_BOW(Mat &img, Mat &clusterCenters, Mat &output);
	int getFeature_HandHOG(Mat &img, Mat &hand, Mat &output);
	int getFeature_HOG_PCA(Mat &img, Mat &output, PCA &pca);
	int getFeature_HOG(Mat &img, Mat &output);
	int getFeature_Shape(Mat &hand, Mat &output);
	int getFeature_Edge(Mat &img, Mat &hand, Mat &output, Mat &output2);

private:
	int computeSift(string workDir, string type);
	int computeContour(string workDir, string type);
	int computeContourPCA(string workDir, string type);
	int computeHOG(string workDir, string type);
	int computeSkeleton(string workDir, string type);	
	int getFeatures(string datasetName, HandInfo &hInfo, Mat &cluster_center_hog, vector<FeatureInfo> &features);
};

Mat getAlignedImg(string imgPath, Rect box, bool needMirror);
Mat getSiftDescriptors(Mat &img);
void visualizeSIFT(Mat &img);
void getHOGDescriptors(Mat &img, Mat &descriptor);
Mat getClusterHOG(string dbName, int version, vector<HandInfo> &handInfo, int objectId=0, int clusterNum=100);
Mat getFeatureMap(const Mat &fvs, const Mat &centers);
int checkHOG(Mat &fv);
int checkSIFT(Mat &fv);


#endif
