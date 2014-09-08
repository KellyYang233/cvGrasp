/*
 *  cvGrasp.hpp
 *  class for unsupervised grasp discovery
 *
 *  Created by Minjie Cai on 10/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_CVGRASP_HPP
#define DEFINE_CVGRASP_HPP

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
#include "FeatureExtractor.hpp"
#include "HiCluster.hpp"
#include "Classifier.hpp"

using namespace std;
using namespace cv;

class cvGrasp
{
public:
	ConfigFile _cfg;
	DataPreparation _dp;
	FeatureExtractor _fe;
	HiCluster _hc;
	string _workDir;
	vector<string> _featureType;

	cvGrasp(string &cfgName);
	int getWorkDir();
	int getFeatureType();
	int prepareData();
	int computeFeature();
	int hierarchyCluster();
	int procIntelGrasp();
	int procGTEAGrasp();
	int procYaleGraspClassify();	
	
private:
	int trainDataGraspClassifier_interval(Dataset &db, int version, string seqName, Grasp &grasp, vector<TrackInfo> &tracks, Mat &data, vector<GraspNode> &nodes);
	int trainDataGraspClassifier(string datasetName, int version, Grasp grasp, vector<Grasp> freqGrasps, Mat &trainData, Mat &labels, vector<GraspNode> &trainNodes);
	template<class T>
	vector<T> trainGraspClassifiers(string datasetName, int version, vector<Grasp> freqGrasps);
	int testDataGraspClassifier_interval(Dataset &db, int version, string seqName, Grasp &grasp, vector<TrackInfo> &tracks, Mat &data, vector<GraspNode> &nodes);
	template<class T>
	pair<Mat, Mat> testGraspClassifier(string datasetName, int version, vector<T> trainers, vector<Grasp> freqGrasps);
	
};

Mat getHOGClusters(Dataset &db, int version, int clusterNum);
Rect getBox_fixed(RotatedRect rRect, Mat &img);
Rect getBox_object(Rect handroi, Mat &img);

#endif
