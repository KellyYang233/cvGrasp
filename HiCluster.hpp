/* 
 *  HiCluster.hpp
 *  HierarchyCluster
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_HICLUSTER_HPP
#define DEFINE_HICLUSTER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "commonUse.hpp"
#include "FeatureExtractor.hpp"
using namespace std;
using namespace cv;

class HiCluster
{
public:
	int _graspNum;
	int _version;	
	string _clusterType;
	vector<float> _weight_set;
	vector<string> _feature_set;
	vector<vector<vector<int> > > _levelTree;
	vector<vector<int> > _minIndex;

	HiCluster() {}
	int initialize(ConfigFile &cfg);
	int cluster(string workDir, vector<string> featureType);
	int getClusterFromIntel(vector<vector<FeatureInfo> > featureInfo);
	int getClusterAllFromIntel(vector<vector<FeatureInfo> > featureInfo);
	int getHiClusterAllFromGTEA(vector<vector<FeatureInfo> > featureInfo);

private:
	int kmeansCluster(string workDir, vector<string> featureType);
	int spectralCluster(string workDir, vector<string> featureType);
	int flatSpectralCluster(string workDir, vector<string> featureType);
	void drawGraspHierarchyHtml(string workDir, string clusterCode, string featCode, int startLevel, int endLevel, bool isMultiple);

};

#endif
