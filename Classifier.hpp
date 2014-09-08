/* 
 *  Classifier.hpp
 *  Supervised Linear Classifier
 *
 *  Created by Minjie Cai on 06/05/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_CLASSIFIER_HPP
#define DEFINE_CLASSIFIER_HPP

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

enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* SVM kernel_type */

/* returns the maximum value of the matrix */
template<class T>
T matrixMaxValue(const Mat& inputMatrix, int& maxY, int& maxX)
{
    cv::Size wh = inputMatrix.size();
    int width = wh.width;
    int height = wh.height;
    T answer = -100000;
    maxY = -1;
    maxX = -1;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            T temp = inputMatrix.at<T>(i,j);
            if(temp >= answer)
            {
                answer = temp;
                maxY = i;
                maxX = j;
            }
        }
    }
    return answer;
}

class LcClassifier
{
public:
	int veb;

	virtual void train(Mat & feature, Mat & labels, string datafile, string modelfile){;}	
	virtual Mat predict(Mat & feature);
	virtual void save( string filename){;}
	virtual void load( string filename){;}
};

class LcSVM : public LcClassifier
{
public:
	int _kernel_type;
	float _gamma;
	Mat _SV;
	Mat _sv_coef;
	Mat _w;
  	float _b;
	float _probA;
	float _probB;

	void train(Mat & feature, Mat & labels, string datafile, string modelfile);
	Mat predict(Mat & feature);
	Mat predict_prob(Mat & feature);
	void save(string filename);
	void load(string filename);

private:
	void write_libsvm_input(const Mat& data, const Mat& labels, const char * filename);
	void read_libsvm_model(const char * filename, int dim);
	Mat RBF_response(Mat &feature);
};

//random trees for regression
class LcRandomTreesR : public LcClassifier
{
public:
	
	CvRTParams _params;
	CvRTrees _random_tree;
	bool _isTrained;

	LcRandomTreesR(){_isTrained = false;}
	void train(Mat & feature, Mat & labels, string datafile, string modelfile);
	Mat predict(Mat & feature);
	Mat predict_prob(Mat & feature);
	void save(string filename);
	void load(string filename);
};

//random trees for classification
class LcRandomTreesC : public LcClassifier
{
public:
	
	CvRTParams _params;
	CvRTrees _random_tree;

	void train(Mat & feature, Mat & labels, string datafile, string modelfile);
	Mat predict(Mat & feature);
	Mat predict_prob(Mat & feature);
	void save(string filename);
	void load(string filename);
};

void write_libsvm_input(const Mat& data, const Mat& labels, const char * filename);
void read_libsvm_model(const char * filename, int dim, LcSVM &model);
Mat classificationToConfusionMatrix(const Mat& classifications, const Mat& labels);

#endif

