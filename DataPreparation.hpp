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

enum GRASP_t
{
	Index_Finger_Extension = 1,
	Large_Diameter = 2,
	Small_Diameter = 3,
	Medium_Wrap = 4,
	Power_Disk = 5,
	Powe_Sphere = 6,
	Adducted_Thumb = 7,
	Light_Tool = 8,
	Fixed_Hook = 9,
	Palmar = 10,
	Platform = 11,
	
	Ring = 12,
	Sphere3_Finger = 13,
	Sphere4_Finger = 14,
	Extension_Type = 15,
	Distal_Type = 16,
	
	Adduction = 17,
	Lateral_Pinch = 18,
	Stick = 19,
	Ventral = 20,
	Lateral_Tripod = 21,
	Tripod_Variation = 22,
	
	ThumbIndex_Finger = 23,
	Tip_Pinch = 24,
	Inferior_Pincer = 25,
	Thumb2_Finger = 26,
	Tripod = 27,
	Thumb3_Finger = 28,
	Quadpod = 29,
	Thumb4_Finger = 30,
	Precision_Disk = 31,
	Precision_Sphere = 32,
	Parallel_Extension = 33,
	
	Writing_Tripod = 34
};

struct Grasp{
	int startFrame;
	int endFrame;
	string graspType;
	void removeBadChars();
};
struct GraspNode{
	string seqName;
	int frameid;
	Rect roi;
	string graspType;
};

#define INTEL_OBJECT_SIZE 42
#define HAND_L 1
#define HAND_R 2
#define HAND_LR 4
#define HAND_ITS 8

struct DatasetSeq{
	string seqName;
	int startFrame;
	int endFrame;
};

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

class Dataset
{
public:
	Dataset(){};
	Dataset(string dbname, string dataDir, int fps, string imType){
		_dbName = dbname;
		_dataDir = dataDir;
		_fps = fps;
		_imType = imType;
		}
	string getDatasetName() const{return _dbName;}
	string getDataDir() const{return _dataDir;}
	int getFps() const{return _fps;}
	string getImType() const{return _imType;}
	void addSequence(const int seqNumber, string seqName, int startFrame, int endFrame){
		DatasetSeq seq;
		seq.seqName = seqName;
		seq.startFrame = startFrame;
		seq.endFrame = endFrame;
		_sequences[seqNumber] = seq;}
	DatasetSeq getSequence(const int seqNumber) {return _sequences[seqNumber];}
	void setTrainSeqs(int version, vector<int> seqs){_trainSeqs[version] = seqs;}
  	vector<int> getTrainSeqs(int version){return _trainSeqs[version];}
	void setTestSeqs(int version, vector<int> seqs){_testSeqs[version] = seqs;}
  	vector<int> getTestSeqs(int version){return _testSeqs[version];}
	vector<Grasp> getGrasps(string seqName);
	vector<Grasp> getfrequentGraspsInTraining(int version, int minFreq);
	vector<Grasp> getfrequentGraspsInTesting(int version, int minFreq);
	vector<Grasp> getFrequentGrasps(int version, int minFreqTraining, int minFreqTesting);
	int getGraspFreqInTraining(Grasp grasp, int version);
	int getGraspFreqInTesting(Grasp grasp, int version);
	vector<Action> getActions(string seqName);
	vector<int> getObjects(string seqName);
	vector<TrackInfo> getTrackedHand(string seqName);
private:
	string _dbName;
	string _dataDir;
	int _fps;
	string _imType;
	map<int, DatasetSeq> _sequences;
	map<int, vector<int> > _trainSeqs;
	map<int, vector<int> > _testSeqs;
};

class DataPreparation
{
public:
	size_t _startframe;
	size_t _endframe;
	size_t _interval;
	string _rootname;
	vector<string> _videoname;
	double _thresPalm;
	int _version;
	map<int, Action> _actions;
	vector<vector<HandInfo> > _handInfo;

	DataPreparation() {}
	int initialize(ConfigFile &cfg);
	int prepare();
	int getGraspFromGTEA_old();
	int getGraspFromGTEA();
	int getGraspFromIntel();
private:	
	int getContours(Mat &src, double thres, vector<vector<Point> > &contours);
	int findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres);
	int getHandRegion(string seqName, int framenum, Mat &anchorPoint);
	int getHandInfo(string seqName, int framenum, TrackInfo &handTrack, HandInfo &hInfo);
	
};

int getContourBig(Mat &prob, Mat &mask, double thres, vector<vector<Point> > &co, int &idx);
vector<Action> readActionAnnotations(const char* filename);
vector<int> readObjectAnnotations(const char* filename);
vector<Grasp> readGraspAnnotations(const char* filename);
vector<TrackInfo> readTrackLog(const char* filename);
Dataset dataset_setup(string datasetName);
Dataset dataset_setup_GTEA(string datasetName);
Dataset dataset_setup_Intel(string datasetName);
Dataset dataset_setup_Yale(string datasetName);

vector<int> randsample(int n, int k);
/* returns only the selected rows in indices */
template<class T>
cv::Mat indMatrixRows(const cv::Mat& M, const std::vector<int>& indices){

  int width = M.cols;
  
  cv::Mat output((int)indices.size(), width, M.type());

  int count = 0;
  for(std::vector<int>::const_iterator it = indices.begin(); it != indices.end(); it++){
    int i = *it;
    for(int j = 0; j < width; j++)
      output.at<T>(count, j) = M.at<T>(i,j);
    count++;
  }

  return output;
}

template<class T>
std::vector<T> indVector(const std::vector<T> &V, const std::vector<int>& indices){
  
  std::vector<T> output((int)indices.size());

  int count = 0;
  for(std::vector<int>::const_iterator it = indices.begin(); it != indices.end(); it++){
    int i = *it;
    output[count] = V[i];
    count++;
  }

  return output;
}

#endif
