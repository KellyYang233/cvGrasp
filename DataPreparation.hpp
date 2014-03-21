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
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "commonUse.hpp"
using namespace std;
using namespace cv;

class DataPreparation
{
public:
	size_t _startframe;
	size_t _endframe;
	size_t _interval;
	string _rootname;
	vector<string> _videoname;
	double _thresPalm;

	DataPreparation() {}
	int initialize(ConfigFile &cfg);
	int prepare();
private:
/*	double getRelevance(vector<Point> contour, int index, double perimeter);
	bool isBlocked(vector<Point> contour, int index);
	void curveEvolution(vector<Point> &contourDCE, const double maxValue, const int minNum);
	void findConvex(vector<Point> list, vector<Point> &convex, vector<Point> &concave);
	void prunConvex(vector<Point> &convex, vector<Point> &concave, const double thres);
	void markContourPartition(vector<Point> contour, vector<Point> convex, Mat &mark);
	void visualizeEndPoint(Mat &img, vector<Point> convex, Rect box);
	void visualizeMark(Mat &img, Mat &mark, vector<Point> contour, Rect box);
	void checkSkeletonPoint(Point center, Mat &dt, Mat &labels, vector<Point> &label_contour, Mat &skeleton, Mat &mark);
	void skeletonize(Mat &dt, Mat &labels, vector<Point> boundary, Point maxLoc, Mat &skeleton);
*/	int getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx);
	int findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres);
};


#endif
