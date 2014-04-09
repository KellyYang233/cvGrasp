/*
 *  DataPreparation.hpp
 *  DataPreparation
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_DATAPREPARATION_HPP
#define DEFINE_DATAPREPARATION_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "commonUse.hpp"
using namespace std;
using namespace cv;

struct Action{
  int startFrame;
  int endFrame;
  int graspFrame;
  string verb;
  vector<string> names;
  void removeBadChars();
  void makeAllCharsSmall();
  void print() const;
  string getActionName() const;
};

vector<Action> readActionAnnotations(const char * filename);

class DataPreparation
{
public:
	size_t _startframe;
	size_t _endframe;
	size_t _interval;
	string _rootname;
	vector<string> _videoname;
	double _thresPalm;
	map<int, Action> _actions;

	DataPreparation() {}
	int initialize(ConfigFile &cfg);
	int prepare();
	int getGraspImgFromAnnotation();
private:
	int getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx);
	int findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres);
	vector<Action> getActions(int seqNumber);
};


#endif
