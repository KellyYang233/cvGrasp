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

#define INTEL_OBJECT_SIZE 42
#define HAND_L 1
#define HAND_R 2
#define HAND_LR 4
#define HAND_ITS 8

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

struct HandInfo{
	int seqNum;
	int frameNum;
	int objectId;
	int handState;
	Rect box[2];
	Point center[2];
	float angle[2];
};

struct TrackInfo{
	int trackNum;
	vector<RotatedRect> rRects;
};

vector<Action> readActionAnnotations(const char* filename);
vector<int> readObjectAnnotations(const char* filename);
vector<TrackInfo> readTrackLog(const char* filename);

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
	vector<vector<HandInfo> > _handInfo;

	DataPreparation() {}
	int initialize(ConfigFile &cfg);
	int prepare();
	int getGraspFromGTEA();
	int getGraspFromIntel();
private:
	int getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx);
	int getContours(Mat &src, double thres, vector<vector<Point> > &contours);
	int findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres);
	int getHandRegion(string seqName, int framenum, Mat &anchorPoint);
	int getHandInfo(string seqName, int framenum, TrackInfo handTrack, HandInfo &hInfo);
	void getActions(string seqName, vector<Action> &seqActions);
	void getObjects(string seqName, vector<int> &objects);
	void getTrackedHand(string seqName, vector<TrackInfo> &tracks);
};


#endif
