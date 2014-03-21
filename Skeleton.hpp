/*
 *  Skeleton.hpp
 *  class for skeleton extraction
 *
 *  Created by Minjie Cai on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_SKELETON_HPP
#define DEFINE_SKELETON_HPP

#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <iomanip>
#include <cstdlib>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Skeleton
{
public:
	bool m_isSkeleton;
        Mat m_prob;
        Mat m_img;
        Mat m_mark;
        Mat m_skeleton;
        vector<Point> m_endpoints;
        vector<vector<float> > m_length;
        vector<vector<vector<float> > > m_radii;

        Skeleton();
        Skeleton(Mat &prob, Mat &img);
        int findSkeleton(double segmentThres, double dceCrit, int dceNum);
        void calcSkeleton();

private:
        Mat m_dt;
        Mat m_labels;
        vector<Point> m_label_contour;
        vector<Point> m_contour;
        vector<Point> m_contourDCE;
        vector<Point> m_convex;
        vector<Point> m_concave;

	int getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx);
        double getRelevance(vector<Point> contour, int index, double perimeter);
        bool isBlocked(vector<Point> contour, int index);
        void curveEvolution(vector<Point> &contourDCE, const double maxValue, const int minNum);
        void findConvex(vector<Point> list, vector<Point> &convex, vector<Point> &concave);
        void prunConvex(vector<Point> &convex, vector<Point> concave, const double thres, vector<Point> contour);
        void markContourPartition(vector<Point> contour, vector<Point> convex, Mat &mark);
        void checkSkeletonPoint(Point center);
        void findPath(vector<Point> endpoints, const int index, vector<vector<Point> > &endpaths);
};

#endif
