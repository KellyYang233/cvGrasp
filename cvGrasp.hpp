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
private:
};

#endif
