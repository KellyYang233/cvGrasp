/*
 *  DataPreparation.cpp
 *  DataPraparation
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <cassert>
#include "DataPreparation.hpp"
#include "Skeleton.hpp"

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

int DataPreparation::findPalm(Mat &p_hand, Point &anchor, Rect &box, Mat &eigenvectors, double segmentThres)
{
	int stat = 0;
        int idx;
        vector<vector<Point> > co;
        Mat binary;
        //check if there is hand with high confidence, and select the biggest contour
        stat = getContourBig(p_hand, binary, segmentThres, co, idx);
	if(stat != 0) return -1;

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
	for(size_t v = 0; v < (size_t)_videoname.size(); v++)
	{
		for(size_t f = _startframe; f <= _endframe; f += _interval)
		{
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
			if(stat) continue;

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
			Mat anchorPoint(1, 10, CV_32FC1);
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
			anchorPoints.push_back(anchorPoint);

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
/*
			// extract skeleton based on connectivity criterion
			Skeleton skeleton(palm, color_roi);
			skeleton.findSkeleton(_thresPalm, 1.0, 10);
	
			//draw skeleton from Harris corner response
			for(int row = 0; row < skeleton.m_skeleton.rows; row++)
				for(int col = 0; col < skeleton.m_skeleton.cols; col++)
				{
					if(skeleton.m_skeleton.at<uchar>(row, col) == 255)
					{
						circle(color_roi, Point(col, row), 1, CV_RGB(0,0,255), -1, CV_AA);
					}
				}
*/			
			ss.str("");
			ss << "sample: " << framenum;
//			putText(color_roi, ss.str(), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255,0,0), 1, CV_AA);
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
