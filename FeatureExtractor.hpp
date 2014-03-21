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
using namespace std;
using namespace cv;

class FeatureExtractor
{
public:
	vector<string> _featureType;
	double _thresContour;

	FeatureExtractor() {}
	int initialize(ConfigFile &cfg);
	int compute(string workDir);

private:
	int computeSift(string workDir, string type);
	int computeContour(string workDir, string type);
	int computeContourPCA(string workDir, string type);
	int computeHOG(string workDir, string type);
	int computeSkeleton(string workDir, string type);
};

#endif
