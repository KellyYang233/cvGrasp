/*
 *  FeatureExtractor.cpp
 *  FeatureExtractor
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <cassert>
#include "FeatureExtractor.hpp"
#include "Skeleton.hpp"

int FeatureExtractor :: initialize(ConfigFile &cfg)
{
	if(cfg.keyExists("feature"))
	{
		string feat = cfg.getValueOfKey<string>("feature");
		while(feat.find(" ") != feat.npos)
		{
			_featureType.push_back(feat.substr(0, feat.find(" ")));
			feat.erase(0, feat.find(" "));
			feat.erase(0, feat.find_first_not_of(" "));
		}
		_featureType.push_back(feat);
	}
	else
	{
		return ERR_CFG_FEATURE_NONEXIST;
	}

	if(cfg.keyExists("thres_contour"))
        {
                _thresContour = cfg.getValueOfKey<double>("thres_contour");
        }
        else
        {
                return ERR_CFG_THRESHOLD_NONEXIST;
        }

	//initilaize surf/sift model
	initModule_nonfree();

	return 0;
}

int FeatureExtractor :: computeHOG(string workDir, string type)
{
	stringstream ss;
        ss << "rm -r " + workDir + "/feature/" + type;
        cout << ss.str() << endl;
        system(ss.str().c_str());

        ss.str("");
        ss << "mkdir " + workDir + "/feature/" + type;
        cout << ss.str() << endl;
        system(ss.str().c_str());

	Size windowSize(80, 80);
	Size blockSize(16, 16);
	Size blockStride(8, 8); Size cellSize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(windowSize, blockSize, blockStride, cellSize, nbins);
	vector<vector<float> >hogVecs;

	Mat anchor;
	ss.str("");
	ss << workDir + "/source/anchor.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[string("anchor")] >> anchor;
	
        for(size_t i = 0; i < MAX_LOOP; i++)
        {
                ss.str("");
                ss << workDir + "/source/img/";
                ss << setw(8) << setfill('0') << i << "_img.jpg";
                Mat img = imread(ss.str());
                if(!img.data && i > 0)
                {
                        cout << "<" << type << "> finish reading img total number: " << i << endl;
                        break;
                }
                if(!img.data && i == 0)
                {
                        return ERR_FEATURE_IMG_NONEXIST;
                }

	/*	Point manp(anchor.at<float>(i, 0), anchor.at<float>(i, 1));
		Point palm_center(anchor.at<float>(i, 2), anchor.at<float>(i, 3));
		Rect roi;
		if(manp.x < palm_center.x && manp.y < palm_center.y)
		{
			roi = Rect(manp.x-40, manp.y-40, 160, 160);
		}
		if(manp.x < palm_center.x && manp.y >= palm_center.y)
		{
			roi = Rect(manp.x-40, manp.y-120, 160, 160);
		}
		if(manp.x >= palm_center.x && manp.y >= palm_center.y)
		{
			roi = Rect(manp.x-120, manp.y-120, 160, 160);
		}
		if(manp.x >= palm_center.x && manp.y < palm_center.y)
		{
			roi = Rect(manp.x-120, manp.y-40, 160, 160);
		}	
		if(roi.x < 0) roi.x = 0;
	   	if(roi.y < 0) roi.y = 0;
                if(roi.x + roi.width > img.cols) roi.x = img.cols - roi.width;
	        if(roi.y + roi.height > img.rows) roi.y = img.rows - roi.height;
	*/
		Rect roi((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat imgroi = img(roi).clone();
		resize(imgroi, imgroi, Size(80,80));
		Mat descriptor;
		vector<float> vec;
		hog.compute(imgroi, vec);
		hogVecs.push_back(vec);
		descriptor = Mat(vec);
		descriptor = descriptor.t();
		//descriptor = descriptor.reshape(0, 1);

		//save HOG descriptor
		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_feature.xml";
		FileStorage fs;
		fs.open(ss.str(), FileStorage::WRITE);
		fs << type << descriptor;
		fs.release();
		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_hog.jpg";
		imwrite(ss.str(), imgroi);
	}

	//compute distance matrix for spectral clustering
	int num_vecs = (int)hogVecs.size();
	int dim = (int)hogVecs[0].size();
	Mat dist_G = Mat::zeros(num_vecs, num_vecs, CV_32FC1);
	for(int i = 0; i < num_vecs; i++)
		for(int j = i+1; j < num_vecs; j++)
		{
			float sum_dist = 0;
			for(int k = 0; k < dim; k++)
				sum_dist += pow(hogVecs[i][k]-hogVecs[j][k],2);
			dist_G.at<float>(i,j) = sqrt(sum_dist/dim);	
		}
	for(int i = 0; i < num_vecs; i++)
		for(int j = 0; j < i; j++)
			dist_G.at<float>(i,j) = dist_G.at<float>(j,i);

	ss.str("");
	ss << workDir + "/feature/" + type + "/feature.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << type << dist_G;
	fs.release();

	return 0;
}

int FeatureExtractor :: computeContourPCA(string workDir, string type)
{
	stringstream ss;
	ss << "rm -r " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	//read anchor information
	Mat anchor;
	ss.str("");
	ss << workDir + "/source/anchor.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[string("anchor")] >> anchor;

	Mat contours;
	for(size_t i = 0; i < MAX_LOOP; i++)
	{
		ss.str("");
		ss << workDir + "/source/hand/";
		ss << setw(8) << setfill('0') << i << "_hand.jpg";
		Mat img = imread(ss.str());
		if(!img.data && i > 0)
		{
			cout << "<" << type << "> finish reading hand probability maps total number: " << i << endl;
			break;
		}
		if(!img.data && i == 0)
		{
			return ERR_FEATURE_HAND_NONEXIST;
		}

		Mat bgr[3];
		Mat p_hand(img.size(), CV_32FC1);
		split(img, bgr);
		for(size_t r = 0; r < (size_t)p_hand.rows; r++)
		{
			for(size_t c = 0; c < (size_t)p_hand.cols; c++)
			{
				p_hand.at<float>(r,c) = bgr[0].at<uchar>(r,c) / 255.0;
				//remove low probability noise, threshold can be set as configure parameter
				if(p_hand.at<float>(r,c) < _thresContour) p_hand.at<float>(r,c) = 0; 
			}
		}

		Rect bounding((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat handroi = p_hand(bounding).clone();
		resize(handroi, handroi, Size(80,100)); 
		handroi = handroi.reshape(0, 1); //convert to one row vector
		contours.push_back(handroi);
	}

	PCA pca(contours, Mat(), CV_PCA_DATA_AS_ROW, 10); //choose 10 principle components
	Mat vecs(contours.rows, 10, contours.type());
	pca.project(contours, vecs);
	for(size_t i = 0; i < (size_t)vecs.rows; i++)
	{
		//save feature to xml
                ss.str("");
                ss << workDir + "/feature/" + type + "/";
                ss << setw(8) << setfill('0') << i << "_feature.xml";
                FileStorage fs;
                fs.open(ss.str(), FileStorage::WRITE);
                fs << type << vecs.row(i);
	}

	//compute distance matrix for spectral clustering
	int num_vecs = vecs.rows;
	int dim = vecs.cols;
	Mat dist_G = Mat::zeros(num_vecs, num_vecs, CV_32FC1);
	for(int i = 0; i < num_vecs; i++)
		for(int j = i+1; j < num_vecs; j++)
		{
			float sum_dist = 0;
			for(int k = 0; k < dim; k++)
				sum_dist += pow(vecs.at<float>(i,k)-vecs.at<float>(j,k), 2);
			dist_G.at<float>(i,j) = sqrt(sum_dist/dim);	
		}
	for(int i = 0; i < num_vecs; i++)
		for(int j = 0; j < i; j++)
			dist_G.at<float>(i,j) = dist_G.at<float>(j,i);

	ss.str("");
	ss << workDir + "/feature/" + type + "/feature.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << type << dist_G;
	fs.release();

	return 0;
}

int FeatureExtractor :: computeContour(string workDir, string type)
{
	stringstream ss;
	ss << "rm -r " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	//read anchor information
	Mat anchor;
	ss.str("");
	ss << workDir + "/source/anchor.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[string("anchor")] >> anchor;

	Mat contourVecs;
	for(size_t i = 0; i < MAX_LOOP; i++)
	{
		ss.str("");
		ss << workDir + "/source/hand/";
		ss << setw(8) << setfill('0') << i << "_hand.jpg";
		Mat img = imread(ss.str());
		if(!img.data && i > 0)
		{
			cout << "<" << type << "> finish reading hand probability maps total number: " << i << endl;
			break;
		}
		if(!img.data && i == 0)
		{
			return ERR_FEATURE_HAND_NONEXIST;
		}

		Mat bgr[3];
		Mat p_hand(img.size(), CV_32FC1);
		split(img, bgr);
		for(size_t r = 0; r < (size_t)p_hand.rows; r++)
		{
			for(size_t c = 0; c < (size_t)p_hand.cols; c++)
			{
				p_hand.at<float>(r,c) = bgr[0].at<uchar>(r,c) / 255.0;
				//remove low probability noise, threshold can be set as configure parameter
				if(p_hand.at<float>(r,c) < _thresContour) p_hand.at<float>(r,c) = 0; 
			}
		}

		Rect bounding((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat handroi = p_hand(bounding).clone();
		resize(handroi, handroi, Size(80,100));
		//save feature to xml
		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_feature.xml";
		FileStorage fs;
		fs.open(ss.str(), FileStorage::WRITE);
		fs << type << handroi;

		//visualize contour image
		double minVal, maxVal;
		minMaxLoc(handroi, &minVal, &maxVal);
		handroi = (handroi-minVal) / (maxVal-minVal);
		handroi *= 255;
		p_hand.convertTo(p_hand, CV_8UC1);
		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_cont.jpg";
		imwrite(ss.str(), handroi);

		handroi = handroi.reshape(0,1);//convert to one row vector
		normalize(handroi, handroi);
		contourVecs.push_back(handroi);
	}

	//compute distance matrix for spectral clustering
	int num_vecs = contourVecs.rows;
	int dim = contourVecs.cols;
	Mat dist_G = Mat::zeros(num_vecs, num_vecs, CV_32FC1);
	for(int i = 0; i < num_vecs; i++)
		for(int j = i+1; j < num_vecs; j++)
		{
			float sum_dist = 0;
			for(int k = 0; k < dim; k++)
				sum_dist += pow(contourVecs.at<float>(i,k)-contourVecs.at<float>(j,k), 2);
			dist_G.at<float>(i,j) = sqrt(sum_dist/dim);	
		}
	for(int i = 0; i < num_vecs; i++)
		for(int j = 0; j < i; j++)
			dist_G.at<float>(i,j) = dist_G.at<float>(j,i);

	ss.str("");
	ss << workDir + "/feature/" + type + "/feature.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << type << dist_G;
	fs.release();

	return 0;
}

int FeatureExtractor :: computeSift(string workDir, string type)
{
	stringstream ss;
	ss << "rm -r " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + workDir + "/feature/" + type;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	//read anchor information
	Mat anchor;
	ss.str("");
	ss << workDir + "/source/anchor.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[string("anchor")] >> anchor;

	vector<KeyPoint> keypoints;
	Mat mask;

	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	if(NULL == detector)
	{
		cout << "fail to create detector" << endl;
		return ERR_FEATURE_DETECTOR_CREAT;
	}

	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	if(NULL == extractor)
	{
		cout << "fail to create extractor" << endl;
		return ERR_FEATURE_EXTRACTOR_CREAT;
	}

	Mat training_descriptors(0, extractor->descriptorSize(), extractor->descriptorType());

	//collect sift descriptors from baselevel images
	for(int i = 0; i < MAX_LOOP; i++)
	{
		ss.str("");
		ss << workDir + "/source/img/";
		ss << setw(8) << setfill('0') << i << "_img.jpg";
		Mat img = imread(ss.str());
		if(!img.data && i > 0)
		{
			cout << "<" << type << "> finish reading images total number: " << i << endl;
			break;
		}
		if(!img.data && i == 0)
		{
			return ERR_FEATURE_IMG_NONEXIST;
		}
		
		Rect roi((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat imgroi = img(roi).clone();
		resize(imgroi, imgroi, Size(80,100));

		detector->detect(imgroi, keypoints, mask);
		Mat descriptors;
		extractor->compute(imgroi, keypoints, descriptors);
		training_descriptors.push_back(descriptors);
	}

	//construct vocabulary of 100 words
	BOWKMeansTrainer bowtrainer(100);
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();
	cout << "<SIFT> finish training vocabulary\n";

	//save vocabulary as xml file
	ss.str("");
	ss << workDir + "/feature/" + type + "/hand_bow.xml";
	FileStorage cvfs(ss.str(), CV_STORAGE_WRITE);
	write(cvfs, "hand_bow", vocabulary);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	//compute image descriptor
	Mat bowVecs;
	for(int i = 0; i < MAX_LOOP; i++)
	{
		ss.str("");
		ss << workDir + "/source/img/";
		ss << setw(8) << setfill('0') << i << "_img.jpg";
		Mat img = imread(ss.str());
		if(!img.data && i > 0)
		{
			cout << "<SIFT> finish computing BOW descriptors total number: " << i << endl;
			break;
		}
		if(!img.data && i == 0)
		{
			return ERR_FEATURE_IMG_NONEXIST;
		}

		Rect roi((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat imgroi = img(roi).clone();
		resize(imgroi, imgroi, Size(80,100));

		Mat imgDescriptor;
		detector->detect(imgroi, keypoints, mask);
		bowide.compute(imgroi, keypoints, imgDescriptor);

		//save sift feature as xml file
		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_feature.xml";
		FileStorage fs;
		fs.open(ss.str(), FileStorage::WRITE);
		fs << string(type) << imgDescriptor;

		for(int j = 0; j < (int)keypoints.size(); j++)
		{
			circle(imgroi, keypoints[j].pt, 2, CV_RGB(0, 255, 0), 1, CV_AA);
			int px = keypoints[j].pt.x + keypoints[j].size/2*cos(keypoints[j].angle*PI/180.0);
			int py = keypoints[j].pt.y + keypoints[j].size/2*sin(keypoints[j].angle*PI/180.0);
			line(imgroi, keypoints[j].pt, Point(px, py), CV_RGB(255, 0, 0), 1, CV_AA);
		}

		ss.str("");
		ss << workDir + "/feature/" + type + "/";
		ss << setw(8) << setfill('0') << i << "_proto.jpg";
		imwrite(ss.str(), imgroi);

		bowVecs.push_back(imgDescriptor);
	}

	//compute distance matrix for spectral clustering
	int num_vecs = bowVecs.rows;
	int dim = bowVecs.cols;
	Mat dist_G = Mat::zeros(num_vecs, num_vecs, CV_32FC1);
	for(int i = 0; i < num_vecs; i++)
		for(int j = i+1; j < num_vecs; j++)
		{
			float sum_dist = 0;
			for(int k = 0; k < dim; k++)
				sum_dist += pow(bowVecs.at<float>(i,k)-bowVecs.at<float>(j,k), 2);
			dist_G.at<float>(i,j) = sqrt(sum_dist/dim);	
		}
	for(int i = 0; i < num_vecs; i++)
		for(int j = 0; j < i; j++)
			dist_G.at<float>(i,j) = dist_G.at<float>(j,i);

	ss.str("");
	ss << workDir + "/feature/" + type + "/feature.xml";
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << type << dist_G;
	fs.release();

	return 0;
}

int matchVertex(vector<vector<float> > &rad_f, vector<float> &len_f, vector<vector<float> > &rad_s, vector<float> &len_s, float &cost_v)
{
	int num_f = (int)rad_f.size();
	int num_s = (int)rad_s.size();
	vector<vector<float> > cost_P(0);
	cost_P.resize(num_f, vector<float>(num_s, 0));

	//calculate path distance
	for(int i = 0; i < num_f; i++)
		for(int j = 0; j < num_s; j++)
		{
			if((int)rad_f[i].size() != (int)rad_s[j].size())
			{
				cout << "ERROR: unequal skeleton samples!\n";
				return -1;
			}
			float cost = 0;
			for(int k = 0; k < (int)rad_f[i].size(); k++)
				cost += pow(rad_f[i][k]-rad_s[j][k], 2) / (rad_f[i][k]+rad_s[j][k]);
			float alpha = 1.0; //weight factor for path length
			cost += alpha * pow(len_f[i]-len_s[j], 2) / (len_f[i]+len_s[j]);
			cost_P[i][j] = cost;
		}

//	print_matrix<float>(cost_P, num_f, num_s);

	//calculate jump cost
	vector<float> minPd(num_f, 0);
	float sum_minPd = 0;
	for(int i = 0; i < num_f; i++)
	{
		float cost_min = FLT_MAX;
		for(int j = 0; j < num_s; j++)
		{
			if(cost_P[i][j] < cost_min)
				cost_min = cost_P[i][j];
		}
		minPd[i] = cost_min;
		sum_minPd += cost_min;
	}
	float mean = sum_minPd / (int)minPd.size();
	float std = 0;
	for(int i = 0; i < (int)minPd.size(); i++)
	{
		std += pow(minPd[i]-mean, 2);
	}
	std = sqrt(std/(int)minPd.size());
	float C = mean + std; //jump cost
//	cout << "jump cost: " << C << "\n";

	//match vertex sequence by using OSB algorithm (preserve sequence order)
	//implement OSB with shortest path algorithm on a DAG
	Graph dag(num_f*num_s);
	for(int i = 0; i < num_f; i++)
		for(int j = 0; j < num_s; j++)
		{
			int u = i*num_s + j;
			for(int k = i+1; k < num_f; k++)
				for(int l = j+1; l < num_s; l++)
				{
					int v = k*num_s + l;
					float weight = sqrt(pow(k-i-1,2) + pow(l-j-1,2)) * C + cost_P[k][l];
					dag.addEdge(u, v, weight);
				}	
		}

	dag.shortestPath(0, num_f*num_s-1, cost_v);
	if(cost_v == FLT_MAX) // in case of one side has only one skeleton path
	{
		cost_v = 0;
		for(int i = 0; i < num_f; i++)
			for(int j = 0; j < num_s; j++)
			{
				cost_v += cost_P[i][j];
			}
	}
	else if(cost_v == 0) // in case of both sides have only one skeleton path
	{
		cost_v = cost_P[0][0];
	}
	else
	{
		cost_v += cost_P[0][0];
	}

	return 0;
}

int matchGraph(vector<vector<vector<float> > > &radii_f, vector<vector<float> > &length_f, vector<Point> &endpoint_f, 
	vector<vector<vector<float> > > &radii_s, vector<vector<float> > &length_s, vector<Point> &endpoint_s, vector<int> &x2y, float &cost_g)
{
	int stat = 0;
	int num_f = (int)radii_f.size();
	int num_s = (int)radii_s.size();
	vector<vector<float> > cost_V(0);
	cost_V.resize(num_f, vector<float>(num_s, 0)); 

	for(int i = 0; i < num_f; i++)
		for(int j = 0; j < num_s; j++)
		{
			float cost = 0;
//			cout << endl;
//			cout << "path distance matrix between endnode[" << i << "] and endnode[" << j << "] >>\n";
			stat = matchVertex(radii_f[i], length_f[i], radii_s[j], length_s[j], cost);
			if(stat)
			{
				cout << "ERROR: call matVertex failed!\n";
				return stat;
			}
			if(cost != 0)
			{
				float rp = sqrt(pow(endpoint_f[i].x-endpoint_s[j].x, 2) + pow(endpoint_f[i].y-endpoint_s[j].y, 2)) / num_f;//penalty for relative position
				cost_V[i][j] = cost + rp;
			}
			else
				cout << "Warning: match two vertex <" << i << ", " << j << "> with zero cost!\n";
		}

	//match two graph using Hungarian algorithm
	vector<vector<int> > input_matrix(0);
	input_matrix.resize(num_f, vector<int>(num_s, 0));
	for(int i = 0; i < num_f; i++)
                for(int j = 0; j < num_s; j++)
		{
			input_matrix[i][j] = (int)(cost_V[i][j]*10);
		}	

	//initialize the gungarian_problem using the cost matrix
	Assignment hungarian(input_matrix, num_f, num_s, HUNGARIAN_MODE_MINIMIZE_COST);

	// some output
//	cout << "endnode dissimilarity matrix:";
//	print_matrix<int>(input_matrix, num_f, num_s);
//        cout << "cost-matrix:";
//        print_matrix<int>(hungarian.org_cost, hungarian.n, hungarian.n);

        // solve the assignement problem
	hungarian.init_labels();
	hungarian.augment();
	x2y = hungarian.xy;
        // some output
//      cout << "assignment:";
	vector<vector<int> >assignment(0);
	assignment.resize(hungarian.n, vector<int>(hungarian.n, 0));
	cost_g = 0;
	for(int i = 0; i < hungarian.n; i++)
	{
		assignment[i][hungarian.xy[i]] = 1;
		cost_g += hungarian.org_cost[i][hungarian.xy[i]];
	}
	cost_g /= 10;
//	print_matrix<int>(assignment, hungarian.n, hungarian.n);
//	cout << "graph matching cost: " << cost_g << endl;
	return 0;
}

int FeatureExtractor :: computeSkeleton(string workDir, string type)
{
        stringstream ss;
        ss << "rm -r " + workDir + "/feature/" + type;
        cout << ss.str() << endl;
        system(ss.str().c_str());

        ss.str("");
        ss << "mkdir " + workDir + "/feature/" + type;
        cout << ss.str() << endl;
        system(ss.str().c_str());

        //read anchor information
        Mat anchor;
        ss.str("");
        ss << workDir + "/source/anchor.xml";
        FileStorage ofs;
        ofs.open(ss.str(), FileStorage::READ);
	if(!ofs.isOpened())
        {
                cout << "ERROR: failed to open file: " << ss.str() << endl;
                return ERR_FEATURE_FILE_OPEN;
        }
        ofs[string("anchor")] >> anchor;
	ofs.release();

	// skeleton map presentation
	vector<Skeleton> skelets;
        for(size_t i = 0; i < MAX_LOOP; i++)
        {
                ss.str("");
                ss << workDir + "/source/hand/";
                ss << setw(8) << setfill('0') << i << "_hand.jpg";
                Mat hand = imread(ss.str());
                if(!hand.data && i > 0)
                {
                        cout << "<" << type << "> finish reading hand probability maps total number: " << i << endl;
                        break;
                }
                if(!hand.data && i == 0)
                {
                        return ERR_FEATURE_HAND_NONEXIST;
                }

                Mat bgr[3];
                Mat p_hand(hand.size(), CV_32FC1);
                split(hand, bgr);
                for(size_t r = 0; r < (size_t)p_hand.rows; r++)
                {
                        for(size_t c = 0; c < (size_t)p_hand.cols; c++)
                        {
                                p_hand.at<float>(r,c) = bgr[0].at<uchar>(r,c) / 255.0;
                                //remove low probability noise, threshold can be set as configure parameter
                                if(p_hand.at<float>(r,c) < _thresContour) p_hand.at<float>(r,c) = 0;
                        }
                }

                ss.str("");
                ss << workDir + "/source/img/";
                ss << setw(8) << setfill('0') << i << "_img.jpg";
                Mat img = imread(ss.str());
                if(!img.data && i > 0)
                {
                        cout << "<" << type << "> finish reading imgs total number: " << i << endl;
                        break;
                }
                if(!img.data && i == 0)
                {
                        return ERR_FEATURE_IMG_NONEXIST;
                }

                Rect roi((int)anchor.at<float>(i,6), (int)anchor.at<float>(i,7), (int)anchor.at<float>(i,8), (int)anchor.at<float>(i,9));
		Mat handroi = p_hand(roi).clone();
                Mat imgroi = img(roi).clone();

		//extract skeleton
		Skeleton skelet(handroi, imgroi);
		skelet.findSkeleton(0.4, 1.0, 10);
		skelet.calcSkeleton();
		skelets.push_back(skelet);
	}

	//prepare for visualizing in html
	ss.str("");
	ss << workDir + "/feature/" + type + "/skeleton-match.html";
	ofstream fs_skeleton;
	fs_skeleton.open(ss.str().c_str());
	if(!fs_skeleton.is_open())
	{
		cout << "ERROR: failed to open file: " << ss.str() << endl;
		return ERR_FEATURE_FILE_OPEN;
	}
	cout << "writing: " << ss.str() << endl;
	fs_skeleton << "<HTML>\n";
        fs_skeleton << "<form name=\"form1\" method=\"post\" action=\"http://rus.hci.iis.u-tokyo.ac.jp/~cai-mj/process_kmeans.php\">\n";
        fs_skeleton << "<H3>skeleton matching pairs [" << (int)skelets.size() << " - " << (int)skelets.size() << "]</H3>\n";
        fs_skeleton << "<br>\n";
        fs_skeleton << "<table border = \"1\" >\n";

	// skeleton matching
	Mat cost_G = Mat::zeros((int)skelets.size(), (int)skelets.size(), CV_32FC1);
	for(int i = 0; i < cost_G.rows; i++)
	{
		fs_skeleton << "    <tr>\n";
		for(int j = i+1; j < cost_G.cols; j++)
		{
			float cost_g;
			vector<int> x2y;
			if(skelets[i].m_isSkeleton == true && skelets[j].m_isSkeleton == true)
			{
				matchGraph(skelets[i].m_radii, skelets[i].m_length, skelets[i].m_endpoints, skelets[j].m_radii, skelets[j].m_length, skelets[j].m_endpoints, x2y, cost_g);
			}
			else
			{
				cost_g = 1000;
			}
			cost_G.at<float>(i, j) = cost_g;

			//visualize skeleton matching
			int maxSize = std::max(std::max(skelets[i].m_img.rows, skelets[i].m_img.cols), std::max(skelets[j].m_img.rows, skelets[j].m_img.cols));
			Mat display = Mat::zeros(maxSize, maxSize*2, CV_8UC3);
			skelets[i].m_img.copyTo(display(Rect(0, 0, skelets[i].m_img.cols, skelets[i].m_img.rows)));
			skelets[j].m_img.copyTo(display(Rect(maxSize, 0, skelets[j].m_img.cols, skelets[j].m_img.rows)));
			if(skelets[i].m_isSkeleton == true && skelets[j].m_isSkeleton == true)
			{
			int num_pairs = std::max((int)skelets[i].m_endpoints.size(), (int)skelets[j].m_endpoints.size());
			if((int)skelets[i].m_endpoints.size() < num_pairs)
			{
				for(int k = 0; k < (int)skelets[i].m_endpoints.size(); k++)
				{
					line(display, skelets[i].m_endpoints[k], Point(skelets[j].m_endpoints[x2y[k]].x+maxSize, skelets[j].m_endpoints[x2y[k]].y), CV_RGB(255,0,0), 2);
				}
			}
			else
			{
				for(int k = 0; k < (int)skelets[i].m_endpoints.size(); k++)
        		        {
					if(x2y[k] >= (int)skelets[j].m_endpoints.size()) continue;
					line(display, skelets[i].m_endpoints[k], Point(skelets[j].m_endpoints[x2y[k]].x+maxSize, skelets[j].m_endpoints[x2y[k]].y), CV_RGB(255,0,0), 2);
        		        }
			}
			}

			ss.str("");
			ss << "match[" << i << "-" << j << "]";
			putText(display, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,0,255), 2, CV_AA);
			ss.str("");
			ss << "cost: " << cost_g;
                        putText(display, ss.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,0,255), 2, CV_AA);
			ss.str("");
                	ss << workDir + "/feature/" + type + "/";
                	ss << "match[" << i << "-" << j << "].jpg";
                	imwrite(ss.str(), display);

			ss.str("");
			ss << "match[" << i << "-" << j << "].jpg";
			fs_skeleton << "        <td>\n";
			fs_skeleton << "        <img src = \"" << ss.str() << "\" WIDTH=\"200\" BORDER=\"0\" >\n";
			fs_skeleton << "        </td>\n";
		}
		fs_skeleton << "    </tr>\n"; 
	}

	//symmentric matching cost
	for(int i = 0; i < cost_G.rows; i++)
		for(int j = 0; j < i; j++)
			cost_G.at<float>(i, j) = cost_G.at<float>(j, i);

	//save feature as xml file
	cout << "save matching cost to xml file\n";
        ss.str("");
        ss << workDir + "/feature/" + type + "/";
        ss << "feature.xml";
        FileStorage ifs;
        ifs.open(ss.str(), FileStorage::WRITE);
	if(!ifs.isOpened())
        {
                cout << "ERROR: failed to open file: " << ss.str() << endl;
                return ERR_FEATURE_FILE_OPEN;
        }
        ifs << string(type) << cost_G;
	ifs.release();

	return 0;
}

int FeatureExtractor :: compute(string workDir)
{
	assert(workDir != "");

	int stat = 0;
	stringstream ss;
	ss << "mkdir " + workDir + "/feature";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	for(size_t f = 0; f < (size_t)_featureType.size(); f++)
	{
		if("SIFT" == _featureType[f])
		{
			stat = computeSift(workDir, _featureType[f]);
			if(stat) return stat;
		}
		else if("CONTOUR" == _featureType[f])
		{
			stat = computeContour(workDir, _featureType[f]);
			if(stat) return stat;
		}
		else if("CONTOURPCA" == _featureType[f])
		{
			stat = computeContourPCA(workDir, _featureType[f]);
			if(stat) return stat;
		}
		else if("HOG" == _featureType[f])
		{
			stat = computeHOG(workDir, _featureType[f]);
			if(stat) return stat;
		}
		else if("SKELETON" == _featureType[f])
		{
			stat = computeSkeleton(workDir, _featureType[f]);
			if(stat) return stat;
		}
		else
		{
			return ERR_CFG_FEATURE_INVALID;
		}
	}

	return 0;
}
