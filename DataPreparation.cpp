/*
 *  DataPreparation.cpp
 *  DataPraparation
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <cassert>
#include <cstdio>
#include "DataPreparation.hpp"

/*****************************************************************************************/
void Action::removeBadChars()
{
	for(int i = 0; i < (int)verb.length(); i++)
		if(verb[i] == ' ' || verb[i] == '-')
			verb[i] = '_';
	for(int o = 0; o < (int)names.size(); o++)
		for(int i = 0; i < (int)names[o].length(); i++)
			if(names[o][i] == ' ' || names[o][i] == '-')
				names[o][i] = '_';
	return;
}

/*****************************************************************************************/
void Action::makeAllCharsSmall()
{
	for(int i = 0; i < (int)verb.length(); i++)
		if(verb[i] >= 'A' && verb[i] <= 'Z')
			verb[i] = verb[i] + 'a' - 'A';
	for(int o = 0; o < (int)names.size(); o++)
		for(int i = 0; i < (int)names[o].length(); i++)
			if(names[o][i] >= 'A' && names[o][i] <= 'Z')
				names[o][i] = names[o][i] + 'a' - 'A';
	return;
}

/*****************************************************************************************/
string Action::getActionName() const
{
	string output("");
	output.append("<");
	output.append(verb);
	output.append("><");
	for(int i = 0; i < (int)names.size()-1; i++)
	{
		output.append(names[i]);
		output.append(",");
	}
	output.append(names[names.size()-1]);
	output.append(">");

	return output;
}

/*****************************************************************************************/
void Action::print() const
{
	cout << getActionName().c_str();
	cout.flush();
	return;
}


/*****************************************************************************************/
vector<Action> readActionAnnotations(const char * filename)
{
	vector<Action> actions;

	ifstream infile(filename, ios::in);
	int limit = 500;
	char line[limit];
	while(infile)
	{
		line[0] = 0;
		infile.getline(line, limit);
		if(!infile)
			break;
		cout << "read line: " << line << endl;
		Action tempAction;
		char * tier = strtok(line, "\t");
		char * startFrame = strtok(NULL, "\t");
		tempAction.startFrame = atoi(startFrame);
		char * endFrame = strtok(NULL, "\t");
		tempAction.endFrame = atoi(endFrame);
		char * action = strtok(NULL, "\t");
		char * graspFrame = strtok(NULL, "\t");
		tempAction.graspFrame = atoi(graspFrame);
		char * actionVerb = strtok(action, "(");
		tempAction.verb = string(actionVerb);
		char * actionNames = strtok(NULL, ")");
cout << "analyze object name: " << actionNames << endl;
		char * name = strtok(actionNames, ",");
		while(name != NULL)
		{
			string namestr(name);
			while(namestr[0] == ' ')
			{
				namestr = namestr.substr(1);
				if(namestr.length() < 1)
					break;
			}
			tempAction.names.push_back(namestr);
			if(tempAction.names.size() >= 3)
				break;
			name = strtok(NULL, ",");
		}
		tempAction.removeBadChars();
		tempAction.makeAllCharsSmall();
		actions.push_back(tempAction);
	}

	infile.close();
	return actions;
}

/*****************************************************************************************/
vector<int> readObjectAnnotations(const char * filename)
{
	vector<int> objects;

	ifstream infile(filename, ios::in);
	int limit = 500;
	char line[limit];
	while(infile)
	{
		line[0] = 0;
		infile.getline(line, limit);
		if(!infile)
			break;
		int objectId = atoi(line);
		objects.push_back(objectId);
	}

	infile.close();
	return objects;
}

int DataPreparation::getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx)
{
	dst = (src > thres)*255;
	if(0 == countNonZero(dst)) return -1;

	vector<Vec4i> hi;
	findContours(dst, co, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	dst *= 0;
	float maxval = -1.0;
	idx = -1;
	for(int c = 0; c < (int)co.size(); c++)
	{
		float area = contourArea(Mat(co[c]));
		if(area < (dst.rows*dst.cols*0.01)) continue;	// too small
		if(area > (dst.rows*dst.cols*0.9)) continue;	// too big
		if(area > maxval)
		{
			maxval = area;
			idx = c;
		}
	}

	if(idx!=-1) drawContours(dst, co, idx, Scalar::all(255.0), -1, CV_AA, hi, 1);
	else return -1;

	return 0;
}

int DataPreparation::getContours(Mat &src, double thres, vector<vector<Point> > &contours)
{
	Mat dst = (src > thres)*255;
	if(0 == countNonZero(dst)) return -1;

	vector<vector<Point> > co;
	vector<Vec4i> hi;
	findContours(dst, co, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	dst *= 0;
	map<float, int> areaRegion;
	for(int c = 0; c < (int)co.size(); c++)
	{
		float area = contourArea(co[c]);
		if(area < (dst.rows*dst.cols*0.02)) continue;	// too small
		if(area > (dst.rows*dst.cols*0.3)) continue;	// too big
		areaRegion[-area] = c;
	}

	count = areaRegion.size();
	if(0 == count)
		return -1;
	for(map<float, int>::iterator it = areaRegion.begin(); it != areaRegion.end(); it++)
		contours.push_back(co[it->second]);

	return 0;
}

int DataPreparation::findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres)
{
	int stat = 0;
	int idx;
	vector<vector<Point> > co;
	Mat binary;
	//check if there is hand with high confidence, and select the biggest contour
	stat = getContourBig(p_hand, binary, segmentThres, co, idx);
	if(stat) return stat;

	//2. calculate mean and covariance matrix for thresholded hand probability map
	Moments moment = moments(binary, true);
	Point dist_mean(moment.m10/moment.m00, moment.m01/moment.m00);
	Mat cov(2, 2, CV_32FC1);
	cov.at<float>(0, 0) = moment.mu20/moment.m00;
	cov.at<float>(0, 1) = moment.mu11/moment.m00;
	cov.at<float>(1, 0) = moment.mu11/moment.m00;
	cov.at<float>(1, 1) = moment.mu02/moment.m00;
	//3. choose eigenvector of covariance matrix with biggest eigenvalue as principle axis
	Mat eigenvalues;
	eigen(cov, eigenvalues, eigenvectors);

	//4. find anchor point using principle axis
	vector<int> hull;
	convexHull(co[idx], hull, true, false);

	double dotMax = FLT_MAX;
	int hullIdx = -1;
	for(int j = 0; j < (int)hull.size(); j++)
	{
		double dotProduct = co[idx][hull[j]].x*eigenvectors.at<float>(0,0) + co[idx][hull[j]].y*eigenvectors.at<float>(0,1);
		if(dotProduct < dotMax)
		{
			dotMax = dotProduct;
			hullIdx = j;
		}
	}
	if(-1 != hullIdx) anchor = co[idx][hull[hullIdx]];
	else
	{
		cout << "ERROR anchor index!\n";
		return -1;
	}

	box = boundingRect(co[idx]);
	return 0;


	//bounding box 240*240 based on anchor point and dist_mean point
	if(anchor.x < dist_mean.x && anchor.y < dist_mean.y)
	{
		box = Rect(anchor.x-80, anchor.y-80, 240, 240);
	}
	if(anchor.x < dist_mean.x && anchor.y >= dist_mean.y)
	{
		box = Rect(anchor.x-80, anchor.y-160, 240, 240);
	}
	if(anchor.x >= dist_mean.x && anchor.y >= dist_mean.y)
	{
		box = Rect(anchor.x-160, anchor.y-160, 240, 240);
	}
	if(anchor.x >= dist_mean.x && anchor.y < dist_mean.y)
	{
		box = Rect(anchor.x-160, anchor.y-80, 240, 240);
	}
	if(box.x < 0) box.x = 0;
	if(box.y < 0) box.y = 0;
	if(box.x + box.width > p_hand.cols) box.x = p_hand.cols - box.width;
	if(box.y + box.height > p_hand.rows) box.y = p_hand.rows - box.height;

	return 0;
}

int DataPreparation :: initialize(ConfigFile &cfg)
{
	if(cfg.keyExists("rootname"))
	{
		_rootname = cfg.getValueOfKey<string>("rootname");
	}
	else
	{
		return ERR_CFG_ROOTNAME_NONEXIST;
	}
	if(cfg.keyExists("videoname"))
	{
		string vidname = cfg.getValueOfKey<string>("videoname");
		while(vidname.find(" ") != vidname.npos)
		{
			_videoname.push_back(vidname.substr(0, vidname.find(" ")));
			vidname.erase(0, vidname.find(" "));
			vidname.erase(0, vidname.find_first_not_of(" "));
		}
		_videoname.push_back(vidname);
	}
	else
	{
		return ERR_CFG_VIDEONAME_NONEXIST;
	}
	if(cfg.keyExists("startframe"))
	{
		_startframe = cfg.getValueOfKey<size_t>("startframe");
	}
	else
	{
		return ERR_CFG_STARTFRAME_NONEXIST;
	}
	if(cfg.keyExists("endframe"))
	{
		_endframe = cfg.getValueOfKey<size_t>("endframe");
	}
	else
	{
		return ERR_CFG_ENDFRAME_NONEXIST;
	}
	if(cfg.keyExists("interval"))
	{
		_interval = cfg.getValueOfKey<size_t>("interval");
	}
	else
	{
		return ERR_CFG_INTERVAL_NONEXIST;
	}

	if(cfg.keyExists("thres_palm"))
	{
		_thresPalm = cfg.getValueOfKey<double>("thres_palm");
	}
	else
	{
		return ERR_CFG_THRESHOLD_NONEXIST;
	}

	//parameter validation
	if(_startframe > _endframe)
	{
		cout << "start frameno should be no more than end frameno\n";
		return ERR_CFG_FRAME_INVALID;
	}

	return 0;
}

int DataPreparation :: prepare()
{
	//initialize() should be called before
	assert((size_t)_videoname.size() > 0);

	int stat;
	string dirCode; //name of base directory
	stringstream ss;
	for(size_t i = 0; i < (size_t)_videoname.size(); i++)
	{
		ss << _videoname[i] << "_";
	}
	dirCode = ss.str();
	dirCode.erase(dirCode.find_last_of("_"));

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "rm -r " + _rootname + "/grasp/" + dirCode + "/source";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source/img";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source/hand";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	Mat anchorPoints;
	int framenum = 0;
	for(int v = 0; v < (int)_videoname.size(); v++)
	{
		for(int f = _startframe; f <= _endframe; f += _interval)
		{
			Mat anchorPoint(1, 10, CV_32FC1);
			stat = getHandRegion(v, f , anchorPoint);
			if(stat)
				continue;
			anchorPoints.push_back(anchorPoint);
			
			//read hand probability image
			ss.str("");
			ss << _rootname + "/hand/" + _videoname[v] + "/";
			ss << setw(8) << setfill('0') << f << ".jpg";
			Mat hand = imread(ss.str());
			if(!hand.data)
			{
				cout << "failed to read hand image: " << ss.str() << endl;
				return ERR_DATA_HAND_NONEXIST;
			}

			//save hand image
			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/hand/";
			ss << setw(8) << setfill('0') << framenum << "_hand.jpg";
			imwrite(ss.str(), hand);
			Mat hand_roi(box.height, box.width, hand.type());
			hand(box).copyTo(hand_roi);
			for(int row = 0; row < hand_roi.rows; row++)
				for(int col = 0; col < hand_roi.cols; col++)
				{
					uchar* ptr = hand_roi.ptr(row);
					*(ptr+col*3+1) = *(ptr+col*3+2) = *(ptr+col*3);
				}

			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/hand/";
			ss << setw(8) << setfill('0') << framenum << "_handroi.jpg";
			imwrite(ss.str(), hand_roi);

			//save color image
			ss.str("");
			ss << _rootname + "/img/" + _videoname[v] + "/" << f << ".jpg";
			Mat color = imread(ss.str());
			if(!color.data)
			{
				return ERR_DATA_IMG_NONEXIST;
			}
			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/img/";
			ss << setw(8) << setfill('0') << framenum << "_img.jpg";
			imwrite(ss.str(), color);
			Mat color_roi(box.height, box.width, color.type());
			color(box).copyTo(color_roi);

			ss.str("");
			ss << "sample: " << framenum;
			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/img/";
			ss << setw(8) << setfill('0') << framenum << "_imgroi.jpg";
			imwrite(ss.str(), color_roi);
			cout << "prepare data num: " << framenum << endl;
			framenum++;
		}
	}

	ss.str("");
	ss << _rootname + "/grasp/" + dirCode + "/source/anchor.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << string("anchor") << anchorPoints;

	return 0;
}

vector<Action> DataPreparation::getActions(string seqName)
{
	vector<Action> seqActions;

	char filename[500];
	filename[0] = 0;
	sprintf(filename, "%s/annotation/T_%s.txt", _rootname.c_str(), seqName.c_str());
	seqActions = readActionAnnotations(filename);

	return seqActions;
}

vector<int> DataPreparation::getObjects(string seqName)
{
	vector<int> objects;

	char filename[500];
	filename[0] = 0;
	sprintf(filename, "%s/annotation/%s.txt", _rootname.c_str(), seqName.c_str());
	objects = readObjectAnnotations(filename);

	return objects;
}

int DataPreparation::getHandRegion(string seqName, int framenum, Mat &anchorPoint)
{
	int stat = 0;
	stringstream ss;
	//read hand probability image
	ss.str("");
	ss << _rootname + "/hand/" + seqName + "/";
	ss << setw(8) << setfill('0') << framenum << ".jpg";
	Mat hand = imread(ss.str());
	if(!hand.data)
	{
		cout << "failed to read hand image: " << ss.str() << endl;
		return ERR_DATA_HAND_NONEXIST;
	}

	Mat bgr[3];
	Mat p_hand(hand.size(), CV_32FC1);
	split(hand, bgr);
	for(size_t r = 0; r < (size_t)p_hand.rows; r++)
	{
		for(size_t c = 0; c < (size_t)p_hand.cols; c++)
		{
			p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
		}
	}

	Point anchor;
	Rect box;
	Mat eigenvectors;
	stat = findPalm(p_hand, anchor, box, eigenvectors, _thresPalm);
	if(stat) return stat;

	//choose search window for palm center and fingertip location
	Mat palm = p_hand(box).clone();
	Mat palm_b(palm.size(), CV_8UC1);

	int idx;
	vector<vector<Point> > co;
	getContourBig(palm, palm_b, _thresPalm, co, idx);
	Rect bounding = boundingRect(co[idx]);
	bounding.x += box.x;
	bounding.y += box.y;
	Mat palm_bb = palm_b.clone();

	// distance transform at search window using second variant of <distanceTransform>
	Mat palm_dt;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	distanceTransform(palm_b, palm_dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	minMaxLoc(palm_dt, &minVal, &maxVal, NULL, &maxLoc);

	//select biggest area after thresholding distance transform map
	getContourBig(palm_dt, palm_b, maxVal/2, co, idx);

	//calculate center of gravity as palm center
	Moments moment = moments(palm_b, true);
	Point palm_center(moment.m10/moment.m00, moment.m01/moment.m00);
	palm_center.x += box.x;
	palm_center.y += box.y;

	//save manipulation points and palm center

	anchorPoint.at<float>(0, 0) = anchor.x;
	anchorPoint.at<float>(0, 1) = anchor.y;
	anchorPoint.at<float>(0, 2) = palm_center.x;
	anchorPoint.at<float>(0, 3) = palm_center.y;
	anchorPoint.at<float>(0, 4) = eigenvectors.at<float>(0,0);
	anchorPoint.at<float>(0, 5) = eigenvectors.at<float>(0,1);
	anchorPoint.at<float>(0, 6) = bounding.x;
	anchorPoint.at<float>(0, 7) = bounding.y;
	anchorPoint.at<float>(0, 8) = bounding.width;
	anchorPoint.at<float>(0, 9) = bounding.height;

	return 0;
}

int DataPreparation::getHandInfo(string seqName, int framenum, HandInfo &hInfo)
{
	int stat = 0;
	stringstream ss;
	//read hand probability image
	ss.str("");
	ss << _rootname + "/hand/" + seqName + "/";
	ss << setw(8) << setfill('0') << framenum << ".jpg";
	Mat hand = imread(ss.str());
	if(!hand.data)
	{
		cout << "failed to read hand image: " << ss.str() << endl;
		return ERR_DATA_HAND_NONEXIST;
	}

	Mat bgr[3];
	Mat p_hand(hand.size(), CV_32FC1);
	split(hand, bgr);
	for(int r = 0; r < p_hand.rows; r++)
	{
		for(int c = 0; c < p_hand.cols; c++)
		{
			p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
		}
	}

	vector<vector<Point> > contours;
	stat = getContours(p_hand, _thresPalm, contours);
	if(stat)
		return ERR_CONTINUE;

	//post-process if needed

	//extract hand information
	if((int)contours.size() == 1)
	{
		RotatedRect rRect = fitEllipse(contours[0]);
		Rect box = boundingRect(contour[0]);
		if(box.width > p_hand.cols/2.0 || box.height > p_hand.rows/2.0)
			hInfo.handState = HAND_ITS;
		else if(rRect.center.x < p_hand.cols/2.0)
			hInfo.handState = HAND_L;
		else
			hInfo.handState = HAND_R;
		hInfo.box[0] = box;
		hInfo.center[0] = rRect.center;
		hInfo.angle[0] = rRect.angle;
	}
	else
	{
		RotatedRect rRect1 = fitEllipse(contours[0]);
		Rect box1 = boundingRect(contour[0]);
		RotatedRect rRect2 = fitEllipse(contours[1]);
		Rect box2 = boundingRect(contour[1]);

		bool isXSec = false;
		bool isYSec = false;
		if(box1.x < box2.x && box1.x+box1.width > box2.x)
			isXSec = true;
		if(box2.x < box1.x && box2.x+box2.width > box1.x)
			isXSec = true;
		if(box1.y < box2.y && box1.y+box1.height > box2.y)
			isYSec = true;
		if(box2.y < box1.y && box2.y+box2.height > box1.y)
			isYSec = true;

		if(isXSec && isYSec)
		{
			hInfo.handState = HAND_ITS;
			int minX = min(box1.x, box2.x);
			int minY = min(box1.y, box2.y);
			int maxX = max(box1.x+box1.width, box2.x+box2.width);
			int maxY = max(box1.y+box1.height, box2.y+box2.height);
			hInfo.box[0] = Rect(minX, minY, maxX-minX, maxY-minY);
			hInfo.center[0] = Point((rRect1.center.x+rRect2.center.x)/2, (rRect1.center.y+rRect2.center.y)/2);
			hInfo.angle[0] = atan((rRect1.center.y-rRect2.center.y)/(rRect1.center.x-rRect2.center.x))*180/PI;
			if(hInfo.angle[0] < 0) hInfo.angle[0] += 180;
		}
		else
		{
			hInfo.handState = HAND_LR;
			if(rRect1.center.x < rRect2.center.x)
			{
				hInfo.box[0] = box1;
				hInfo.center[0] = rRect1.center;
				hInfo.angle[0] = rRect1.angle;
				hInfo.box[1] = box2;
				hInfo.center[1] = rRect2.center;
				hInfo.angle[1] = rRect2.angle;
			}
			else
			{
				hInfo.box[0] = box2;
				hInfo.center[0] = rRect2.center;
				hInfo.angle[0] = rRect2.angle;
				hInfo.box[1] = box1;
				hInfo.center[1] = rRect1.center;
				hInfo.angle[1] = rRect1.angle;
			}
		}		
			
	}
	
}

int DataPreparation::getGraspFromGTEA()
{
	//initialize() should be called before
	assert((size_t)_videoname.size() > 0);

	int stat;
	string dirCode; //name of base directory
	stringstream ss;
	for(size_t i = 0; i < (size_t)_videoname.size(); i++)
	{
		ss << _videoname[i] << "_";
	}
	dirCode = ss.str();
	dirCode.erase(dirCode.find_last_of("_"));

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "rm -r " + _rootname + "/grasp/" + dirCode + "/source";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source/img";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/source/hand";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	Mat anchorPoints;
	ss.str("");
	ss << _rootname + "/grasp/" + dirCode + "/source/frameid.txt";
	ofstream outfile(ss.str().c_str());
	int framenum = 0;
	for(int v = 0; v < (int)_videoname.size(); v++)
	{
		vector<Action> seqActions = getActions(_videoname[v]);
		cout << "read action annotation in sequence: " << _videoname[v] << endl;
		for(int a = 0; a < (int)seqActions.size(); a++)
		{
			int frameid = seqActions[a].graspFrame;
			Mat anchorPoint(1, 10, CV_32FC1);
			stat = getHandRegion(_videoname[v], frameid , anchorPoint);
			if(stat)
				continue;
			anchorPoints.push_back(anchorPoint);
			_actions[framenum] = seqActions[a];
			ss.str("");
			ss << _videoname[v] << "\t" << frameid << "\t" << framenum << endl;
			outfile << ss.str();

			//read hand probability image
			ss.str("");
			ss << _rootname + "/hand/" + _videoname[v] + "/";
			ss << setw(8) << setfill('0') << frameid << ".jpg";
			Mat hand = imread(ss.str());
			if(!hand.data)
			{
				cout << "failed to read hand image: " << ss.str() << endl;
				return ERR_DATA_HAND_NONEXIST;
			}
			//save hand image
			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/hand/";
			ss << setw(8) << setfill('0') << framenum << "_hand.jpg";
			imwrite(ss.str(), hand);

			Rect box(anchorPoint.at<float>(0, 6), anchorPoint.at<float>(0, 7), anchorPoint.at<float>(0, 8), anchorPoint.at<float>(0, 9));
			Mat hand_roi(box.height, box.width, hand.type());
			hand(box).copyTo(hand_roi);
			for(int row = 0; row < hand_roi.rows; row++)
				for(int col = 0; col < hand_roi.cols; col++)
				{
					uchar* ptr = hand_roi.ptr(row);
					*(ptr+col*3+1) = *(ptr+col*3+2) = *(ptr+col*3);
				}

			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/hand/";
			ss << setw(8) << setfill('0') << framenum << "_handroi.jpg";
			imwrite(ss.str(), hand_roi);

			//save color image
			ss.str("");
			ss << _rootname + "/img/" + _videoname[v] + "/" << frameid << ".jpg";
			Mat color = imread(ss.str());
			if(!color.data)
			{
				return ERR_DATA_IMG_NONEXIST;
			}
			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/img/";
			ss << setw(8) << setfill('0') << framenum << "_img.jpg";
			imwrite(ss.str(), color);
			Mat color_roi(box.height, box.width, color.type());
			color(box).copyTo(color_roi);

			ss.str("");
			ss << _rootname + "/grasp/" + dirCode + "/source/img/";
			ss << setw(8) << setfill('0') << framenum << "_imgroi.jpg";
			imwrite(ss.str(), color_roi);
			cout << "prepare data num: " << framenum << endl;
			framenum++;
		}
	}
	ss.str("");
	ss << _rootname + "/grasp/" + dirCode + "/source/anchor.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << string("anchor") << anchorPoints;
	outfile.close();

	return 0;
}

int DataPreparation::getGraspFromIntel()
{
	int stat;
	string dirCode = "Egocentric_Objects_Intel"; //name of base directory
	stringstream ss;

	ss.str("");
	ss << "mkdir " + _rootname + "/grasp/" + dirCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	vector<int> trainS;
	trainS.push_back(3);
	trainS.push_back(4);
//	trainS.push_back(6);
//	trainS.push_back(8);
//	trainS.push_back(10);

	_handInfo = vector<vector<HandInfo> >(INTEL_OBJECT_SIZE);

	for(int s = 0;  s < (int)trainS.size(); s++)
	{
		ss.str("");
		ss << "Egocentric_Objects_Intel/no" << trainS[s];
		vector<int> objectList = getObjects(ss.str());
		for(int f = 0; f < (int)objectList.size(); f++)
		{
			HandInfo hInfo;
			hInfo.seqNum = trainS[s];
			hInfo.frameNum = f+1;
			if(objectList[f] == 0)
				continue;
			hInfo.objectId = objectList[f];
			stat = getHandInfo(ss.str(), f+1, hInfo);
			if(stat)
			{
				if(stat == ERR_CONTINUE)
					continue;
				else
					break;
			}
			
			_handInfo[hInfo.objectId-1].push_back(hInfo);
		}
	}

	for(int i = 0; i < INTEL_OBJECT_SIZE; i++)
	{
		ss.str("");
		ss << "rm -r " + _rootname + "/grasp/" + dirCode + "/object-" << i+1;
		cout << ss.str() << endl;
		system(ss.str().c_str());

		ss.str("");
		ss << "mkdir " + _rootname + "/grasp/" + dirCode + "/object-" << i+1;
		cout << ss.str() << endl;
		system(ss.str().c_str());

		for(int j = 0; j < (int)_handInfo[i].size(); j++)
		{
			ss.str("");
			ss << _rootname + "/img/Egocentric_Objects_Intel/no" << _handInfo[i][j].seqNum + "/";
			ss << setw(10) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
			Mat img = imread(ss.str());
			if(!img.data)
			{
				cout << "failed to read color image: " << ss.str() << endl;
				return ERR_DATA_IMG_NONEXIST;
			}

			//save color image
			if(_handInfo[i][j].handState == HAND_L)
			{
				Mat img_roi(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
				img(_handInfo[i][j].box[0]).copyTo(img_roi);

				ss.str("");
				ss << _rootname + "/grasp/" + dirCode + "/object-" << i+1 << "/L_";
				ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
				imwrite(ss.str(), img_roi);
			}
			else if(_handInfo[i][j].handState == HAND_R)
			{
				Mat img_roi(_handInfo[i][j].box[1].height, _handInfo[i][j].box[1].width, img.type());
				img(_handInfo[i][j].box[1]).copyTo(img_roi);

				ss.str("");
				ss << _rootname + "/grasp/" + dirCode + "/object-" << i+1 << "/R_";
				ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
				imwrite(ss.str(), img_roi);
			}
			else if(_handInfo[i][j].handState == HAND_LR)
			{
				Mat img_roi1(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
				img(_handInfo[i][j].box[0]).copyTo(img_roi1);

				ss.str("");
				ss << _rootname + "/grasp/" + dirCode + "/object-" << i+1 << "/L_";
				ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
				imwrite(ss.str(), img_roi1);

				Mat img_roi2(_handInfo[i][j].box[1].height, _handInfo[i][j].box[1].width, img.type());
				img(_handInfo[i][j].box[1]).copyTo(img_roi);

				ss.str("");
				ss << _rootname + "/grasp/" + dirCode + "/object-" << i+1 << "/R_";
				ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
				imwrite(ss.str(), img_roi2);
			}
			else if(_handInfo[i][j].handState == HAND_ITS)
			{
				Mat img_roi(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
				img(_handInfo[i][j].box[0]).copyTo(img_roi);

				ss.str("");
				ss << _rootname + "/grasp/" + dirCode + "/object-" << i+1 << "/ITS";
				ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
				imwrite(ss.str(), img_roi);
			}
			else
			{
				cout << "bad hand state at sequence " << _handInfo[i][j].seqNum << " frame " << _handInfo[i][j].frameNum << endl;
				continue;
			}
			
		}
	}
}

