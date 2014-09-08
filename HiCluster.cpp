/*
 *  HiCluster.cpp
 *  HierarchyCluster
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <cassert>
#include "HiCluster.hpp"

float calcDissimilarity(FeatureInfo &F1, FeatureInfo &F2, vector<string> feature_set, vector<float> weight_set)
{
	float cost = 0;

	for(int i = 0; i < (int)feature_set.size(); i++)
	{
		if("HOG" == feature_set[i])
		{
			cost += norm(F1.hog, F2.hog)*weight_set[i];
		}
		else if("EDGE" == feature_set[i])
		{
			if(F1.edge.size() != F2.edge.size())
			{
				cout << "calcDissimilarity: ERROR: size of edge map should be equal\n";
				exit(1);
			}
			float sum1=0, sum2=0;
			int c1=0, c2=0;
			for(int r = 0; r < F1.edge.rows; r++)
				for(int c = 0; c < F1.edge.cols; c++)
				{
					if(F1.edge.at<uchar>(r,c) > 0)
					{
						sum1 += F2.dt.at<float>(r,c);
						c1++;
					}
					if(F2.edge.at<uchar>(r,c) > 0)
					{
						sum2 += F1.dt.at<float>(r,c);
						c2++;
					}
				}
			cost += (sum1/c1 + sum2/c2)*weight_set[i];
					
		}
		else if("SHAPE" == feature_set[i])
		{
			cost += norm(F1.shape, F2.shape)*weight_set[i];
		}
		else
		{
			cout << "calcDissimilarity: WARNING: unrecognized feature type: " + feature_set[i] + "\n";
		}
	}
	
	return cost;
}

void getSpectralCluster(Mat &distMatrix, int K, Mat &labels, Mat &centers)
{
	assert(K > 1 && distMatrix.data);
	//similarity matrix
	Mat S = Mat::zeros(distMatrix.size(), CV_32FC1); 
	for(int row = 0; row < S.rows; row++)
	{
		float sum_row = 0;
		for(int k = 0; k < S.cols; k++)
			sum_row += distMatrix.at<float>(row, k);
		float std = sum_row / S.cols;
//		cout << "standard deviation <row" << row << ">: " << std << endl;	
		for(int col = row+1; col < S.cols; col++)
		{
			S.at<float>(row, col) = exp(pow(distMatrix.at<float>(row,col),2)*(-1)/2/pow(std,2));
		}
	}
	for(int row = 0; row < S.rows; row++)
		for(int col = 0; col < row; col++)
			S.at<float>(row, col) = S.at<float>(col, row);

	//diagonal matrix
	Mat D = Mat::zeros(S.size(), CV_32FC1);
	for(int i = 0; i < S.rows; i++)
	{
		float sum_row = 0;
		for(int j = 0; j < S.cols; j++)
		{
			sum_row += S.at<float>(i,j);
		}
		D.at<float>(i, i) = 1 / sqrt(sum_row);
	}

	//normalized graph Laplacian
	Mat L = D * S * D;
	Mat eigenvectors;
	vector<float> eigenvalues;
	eigen(L, eigenvalues, eigenvectors);
/*	for(int i = 0; i < (int)eigenvalues.size(); i++)
	{
		cout << eigenvalues[i] << " ";
		if(i%10 == 9) cout << endl;
	}
*/
	Mat samples(L.rows, K, CV_32FC1);
	eigenvectors = eigenvectors.t();
	eigenvectors.colRange(0, K).copyTo(samples);
	for(int i = 0; i < samples.rows; i++)
		normalize(samples.row(i), samples.row(i));

	//kmeans clustering
	kmeans(samples, K, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.1/K), 5, KMEANS_PP_CENTERS, centers);

	return;
}

int HiCluster :: initialize(ConfigFile &cfg)
{
	if(cfg.keyExists("cluster"))
	{
		_clusterType = cfg.getValueOfKey<string>("cluster");
	}
	else
	{
		return ERR_CFG_CLUSTER_NONEXIST;
	}

	if(cfg.keyExists("version"))
	{
		_version = cfg.getValueOfKey<int>("version");
	}
	else
	{
		return ERR_CFG_VERSION_NONEXIST;
	}

	if(cfg.keyExists("feature"))
    {
        string fCode = cfg.getValueOfKey<string>("feature");
		cout << "features: " << fCode << endl;
		//parse feature code and calculate corresponding feature cost
		//element of feature code: (type, weight)
		char* type = strtok((char*)fCode.c_str(), " ,;");
		while(type != NULL)
		{
			_feature_set.push_back(string(type));
			char* weight = strtok(NULL, " ,;");
			if(weight == NULL)
			{
				cout << "HiCluster::initialize: WARNING: invalid feature configuration\n";
				_feature_set.pop_back();
				break;
			}
			_weight_set.push_back(atof(weight));
			type = strtok(NULL, " ,;");
		}
    }

	return 0;
}

int HiCluster :: kmeansCluster(string workDir, vector<string> featureType)
{
	int stat = 0;
	stringstream ss;
	ss.str("");
	ss << "mkdir " + workDir + "/cluster/kmeans";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	for(size_t i = 0; i < (size_t)featureType.size(); i++)
	{
		ss << featureType[i] << "_";
	}
	string featCode = ss.str();
	featCode.erase(featCode.find_last_of("_"));

	ss.str("");
	ss << "rm -r " + workDir + "/cluster/kmeans/" + featCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + workDir + "/cluster/kmeans/" + featCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	size_t baselevelSize = 0;
	//baselevel feature normalization and combination
	for(size_t i = 0; i < MAX_LOOP; i++)
	{
		Mat desc;
		for(size_t f = 0; f < (size_t)featureType.size(); f++)
		{
			Mat feat;
			ss.str("");
			ss << workDir + "/feature/" + featureType[f] + "/";
			ss << setw(8) << setfill('0') << i << "_feature.xml";
			FileStorage ofs;
			ofs.open(ss.str(), FileStorage::READ);
			ofs[featureType[f]] >> feat;
			if(!feat.data && i > 0)
			{
				stat = 1;
				cout << "<kmeans> finish reading " << featCode << " features, total number: " << i << endl;
				break;
			}
			if(!feat.data && i == 0)
			{
				return ERR_CLUSTER_FEATURE_NONEXIST;
			}
			//normalize the feature, we should add weight for different feature types later
			normalize(feat, feat);
			feat = feat.t(); //transpose for later combination
			if( (desc.data) && (feat.cols != desc.cols) )
			{
				cout << "ERROR: Cols of feature " << featureType[f] << " don't match\n";
				return ERR_CLUSTER_FEATURE_DIMENSION;
			}
			desc.push_back(feat);
		}
		if(stat)
		{
			stat = 0;
			baselevelSize = i;
			break;
		}

		//transpose combined features as row vector
		desc = desc.t();

		ss.str("");
		ss << workDir + "/cluster/kmeans/" + featCode + "/level_0_" << i << "_feature.xml";
		FileStorage ifs;
		ifs.open(ss.str(), FileStorage::WRITE);
		ifs << featCode << desc;
	}

	//structure for storing prototype index
	_minIndex.push_back(vector<int>());
	for(size_t i = 0; i < baselevelSize; i++)
	{
		_minIndex[0].push_back(i);
	}

	//hierarchical clustering levels, with 1/2 convergence rate
	size_t totalLevel = floor( log2(baselevelSize) );
	// clustering centers of each group using k-means
	for(size_t lev = 1; lev < totalLevel; lev++)
	{
		size_t rows_prev;
		size_t clusterCount = baselevelSize >> lev;
		Mat samples;

		for(size_t i = 0; i < MAX_LOOP; i++)
		{
			ss.str("");
			ss << workDir + "/cluster/kmeans/" + featCode + "/level_" << lev-1 << "_" << i << "_feature.xml" ;
			Mat sample;
			FileStorage fs;
			fs.open(ss.str(), FileStorage::READ);
			fs[featCode] >> sample;
			if(!sample.data && i > 0)
			{
				cout << "finish reading feature from level: " << lev-1 << " total number: " << i << endl;
				break;
			}
			if(!sample.data && i == 0)
			{
				return ERR_CLUSTER_FEATURE_NONEXIST;
			}
			rows_prev = sample.rows;
			sample.convertTo(sample, CV_32FC1);
			sample = sample.reshape(0, 1); // convert to one-row vector
			samples.push_back(sample);
		}

		//kmeans clustering
		Mat centers, labels;
		kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
		cout << "finish kmeans clustering for level: " << lev << endl;

		//restore cluster relationship of current level
		vector<vector<int> > clusterTree(centers.rows);
		for(size_t k = 0; k < (size_t)samples.rows; k++)
		{
			clusterTree[labels.at<int>(k, 0)].push_back(k);
		}
		_levelTree.push_back(clusterTree);


		//train knn with kmeans cluster centers
		CvKNearest knn;
		for(size_t k = 0; k < (size_t)centers.rows; k++)
		{
			Mat center(1, centers.cols, centers.type());
			centers.row(k).copyTo(center);
			Mat label = Mat::ones(1, 1, CV_32F)*k;
			if(k == 0)
			{
				knn.train(center, label, Mat(), false, 1, false);
			}
			else
			{
				knn.train(center, label, Mat(), false, 1, true);
			}

		}

		//find the nearest grasp for each cluster center from base level grasps
		Mat sams;
		Mat lab, rsp, dis;
		for(size_t i = 0; i < MAX_LOOP; i++)
		{
			ss.str("");
			ss << workDir + "/cluster/kmeans/" + featCode + "/level_" << 0 << "_" << i << "_feature.xml" ;
			Mat sample;
			FileStorage fs;
			fs.open(ss.str(), FileStorage::READ);
			fs[featCode] >> sample;
			if(!sample.data) break;

			sample.convertTo(sample, CV_32FC1);
			sample = sample.reshape(0, 1); // convert to one-row vector
			sams.push_back(sample);
		}
		knn.find_nearest(sams, 1, lab, rsp, dis);

		vector<float> minVal(centers.rows, INFINITY);
		vector<int> minIndex(centers.rows, -1);
		for(size_t i = 0; i < (size_t)sams.rows; i++)
		{
			int cluster_id = lab.at<float>(i, 0);
			if(dis.at<float>(i, 0) < minVal[cluster_id] && dis.at<float>(i, 0) >=0)
			{
				minVal[cluster_id] = dis.at<float>(i, 0);
				minIndex[cluster_id] = i;
			}
		}

		//restore the nearest neighbor grasp for later visualization
		_minIndex.push_back(minIndex);

		//save cluster centers as new level features
		for(size_t i = 0; i < (size_t)centers.rows; i++)
		{
			Mat center(1, centers.cols, centers.type());
			centers.row(i).copyTo(center);
			center.convertTo(center, CV_32FC1);
			center = center.reshape(0, rows_prev);

			// write binary image of cluster centers
			ss.str("");
			ss << workDir + "/cluster/kmeans/" + featCode + "/level_" << lev << "_" << i << "_feature.xml" ;
			FileStorage fs;
			fs.open(ss.str(), FileStorage::WRITE);
			fs << featCode << center;

			if("CONTOUR" == featCode)
			{
				double maxVal, minVal;
				minMaxLoc(center, &minVal, &maxVal);
				center = (center-minVal) / (maxVal-minVal);
				center *= 255;
				center.convertTo(center, CV_8UC1);
				ss.str("");
				ss << workDir + "/cluster/kmeans/" + featCode + "/level_" << lev << "_" << i << "_cont.jpg" ;
				imwrite(ss.str(), center);
			}
		}
	}

	_levelTree.push_back(vector< vector<int> >());
	_levelTree[totalLevel-1].push_back(vector<int>(1,0));
	_levelTree[totalLevel-1].push_back(vector<int>(1,1));
	if(baselevelSize >> (totalLevel-1) >= 3) _levelTree[totalLevel-1].push_back(vector<int>(1,2));

	//save hierarchical tree structure to html
	drawGraspHierarchyHtml(workDir, string("kmeans"), featCode, 0, totalLevel-1, false);
	drawGraspHierarchyHtml(workDir, string("kmeans"), featCode, 0, totalLevel-3, true);
	return 0;
}

int HiCluster :: flatSpectralCluster(string workDir, vector<string> featureType)
{
	stringstream ss;
	ss.str("");
	ss << "mkdir " + workDir + "/cluster/spectral";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	for(size_t i = 0; i < (size_t)featureType.size(); i++)
	{
		ss << featureType[i] << "_";
	}
	string featCode = ss.str();
	featCode.erase(featCode.find_last_of("_"));
	/*
		ss.str("");
		ss << "rm -r " + workDir + "/cluster/spectral/" + featCode;
		cout << ss.str() << endl;
		system(ss.str().c_str());
	*/
	ss.str("");
	ss << "mkdir " + workDir + "/cluster/spectral/" + featCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	_minIndex = vector<vector<int> >(0);
	_levelTree = vector<vector<vector<int> > >(0);

	Mat cost_G; //distance matrix
	ss.str("");
	ss << workDir + "/feature/" + featCode + "/feature.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[featCode] >> cost_G;
	if(!cost_G.data)
	{
		return ERR_CLUSTER_FEATURE_NONEXIST;
	}


	int baselevelSize = cost_G.rows;
	//structure for storing prototype index
	_minIndex.push_back(vector<int>());
	for(int i = 0; i < baselevelSize; i++)
	{
		_minIndex[0].push_back(i);
	}

//	float std = 10; //strongly recommend to set it adaptative to data distribution
	Mat S = Mat::zeros(cost_G.size(), CV_32FC1); //similarity matrix
	for(int row = 0; row < S.rows; row++)
	{
		float sum_row = 0;
		for(int k = 0; k < S.cols; k++)
			sum_row += cost_G.at<float>(row, k);
		float std = sum_row / S.cols;
		cout << "standard deviation <row" << row << ">: " << std << endl;	
		for(int col = row+1; col < S.cols; col++)
		{
			S.at<float>(row, col) = exp(pow(cost_G.at<float>(row,col),2)*(-1)/2/pow(std,2));
		}
	}
	for(int row = 0; row < S.rows; row++)
		for(int col = 0; col < row; col++)
			S.at<float>(row, col) = S.at<float>(col, row);

	//diagonal matrix
	Mat D = Mat::zeros(S.size(), CV_32FC1);
	for(int i = 0; i < S.rows; i++)
	{
		float sum_row = 0;
		for(int j = 0; j < S.cols; j++)
		{
			sum_row += S.at<float>(i,j);
		}
		D.at<float>(i, i) = 1 / sqrt(sum_row);
	}

	//graph Laplacian
	Mat L = D * S * D;
	Mat eigenvectors;
	vector<float> eigenvalues;
	eigen(L, eigenvalues, eigenvectors);
	for(int i = 0; i < (int)eigenvalues.size(); i++)
	{
		cout << eigenvalues[i] << " ";
		if(i%10 == 9) cout << endl;
	}

	int clusterCount = 7;
	Mat samples;
	eigenvectors = eigenvectors.t();
	eigenvectors.colRange(0, clusterCount-1).copyTo(samples);
	for(int i = 0; i < samples.rows; i++)
		normalize(samples.row(i), samples.row(i));

	//kmeans clustering
	Mat centers, labels;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
	cout << "\nfinish flat spectral clustering" << endl;

	//restore cluster relationship of current level
	vector<vector<int> > clusterTree(centers.rows);
	for(int k = 0; k < samples.rows; k++)
	{
		clusterTree[labels.at<int>(k, 0)].push_back(k);
	}
	_levelTree.push_back(clusterTree);


	//train knn with kmeans cluster centers
	CvKNearest knn;
	for(size_t k = 0; k < (size_t)centers.rows; k++)
	{
		Mat center(1, centers.cols, centers.type());
		centers.row(k).copyTo(center);
		Mat label = Mat::ones(1, 1, CV_32F)*k;
		if(k == 0)
		{
			knn.train(center, label, Mat(), false, 1, false);
		}
		else
		{
			knn.train(center, label, Mat(), false, 1, true);
		}

	}

	//find the nearest grasp for each cluster center from current level
	Mat lab, rsp, dis;
	knn.find_nearest(samples, 1, lab, rsp, dis);

	vector<float> minVal(centers.rows, INFINITY);
	vector<int> minIndex(centers.rows, -1);
	for(size_t i = 0; i < (size_t)samples.rows; i++)
	{
		int cluster_id = lab.at<float>(i, 0);
		if(dis.at<float>(i, 0) < minVal[cluster_id] && dis.at<float>(i, 0) >=0)
		{
			minVal[cluster_id] = dis.at<float>(i, 0);
			minIndex[cluster_id] = _minIndex[0][i];
		}
	}

	//restore the nearest neighbor grasp for later visualization
	_minIndex.push_back(minIndex);

	//save average similarity between current clusters as new level features
	Mat C_new = Mat::zeros(centers.rows, centers.rows, CV_32FC1);
	for(int i = 0; i < centers.rows; i++)
		for(int j = i+1; j < centers.rows; j++)
		{
			float sum_cost = 0;
			for(int k = 0; k < (int)clusterTree[i].size(); k++)
				for(int l = 0; l < (int)clusterTree[j].size(); l++)
					sum_cost += cost_G.at<float>(clusterTree[i][k], clusterTree[j][l]);
			C_new.at<float>(i, j) = sum_cost / (int)clusterTree[i].size() / (int)clusterTree[j].size();
		}
	for(int i = 0; i < C_new.rows; i++)
		for(int j = 0; j < i; j++)
			C_new.at<float>(i, j) = C_new.at<float>(j, i);

	// write binary image of cluster centers
	ss.str("");
	ss << workDir + "/cluster/spectral/" + featCode + "/flat_feature.xml" ;
	FileStorage fs;
	fs.open(ss.str(), FileStorage::WRITE);
	fs << featCode << C_new;
	fs.release();

	_levelTree.push_back(vector< vector<int> >());
	for(int i = 0; i < clusterCount; i++)
		_levelTree[1].push_back(vector<int>(1,i));

	//save hierarchical tree structure to html
	drawGraspHierarchyHtml(workDir, string("spectral"), featCode, 0, 1, true);

	return 0;
}

int HiCluster :: spectralCluster(string workDir, vector<string> featureType)
{
	stringstream ss;
	ss.str("");
	ss << "mkdir " + workDir + "/cluster/spectral";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	for(size_t i = 0; i < (size_t)featureType.size(); i++)
	{
		ss << featureType[i] << "_";
	}
	string featCode = ss.str();
	featCode.erase(featCode.find_last_of("_"));

	ss.str("");
	ss << "rm -r " + workDir + "/cluster/spectral/" + featCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "mkdir " + workDir + "/cluster/spectral/" + featCode;
	cout << ss.str() << endl;
	system(ss.str().c_str());

	_minIndex = vector<vector<int> >(0);
	_levelTree = vector<vector<vector<int> > >(0);
	Mat cost_G; //similarity matrix
	ss.str("");
	ss << workDir + "/feature/" + featCode + "/feature.xml";
	FileStorage ofs;
	ofs.open(ss.str(), FileStorage::READ);
	ofs[featCode] >> cost_G;
	if(!cost_G.data)
	{
		return ERR_CLUSTER_FEATURE_NONEXIST;
	}

	cout << "<" << featCode << "> save baselevel distance matrix, baselevel size: " << cost_G.rows << endl;
	ss.str("");
	ss << workDir + "/cluster/spectral/" + featCode + "/level_0_feature.xml";
	FileStorage ifs;
	ifs.open(ss.str(), FileStorage::WRITE);
	ifs << featCode << cost_G;
	ifs.release();

	int baselevelSize = cost_G.rows;
	//structure for storing prototype index
	_minIndex.push_back(vector<int>());
	for(int i = 0; i < baselevelSize; i++)
	{
		_minIndex[0].push_back(i);
	}

	//hierarchical clustering levels, with 1/2 convergence rate
	int totalLevel = floor( log2(baselevelSize) );
	// clustering centers of each group using k-means
	for(int lev = 1; lev < totalLevel; lev++)
	{
		int clusterCount = baselevelSize >> lev;
		Mat samples;

		ss.str("");
		ss << workDir + "/cluster/spectral/" + featCode + "/level_" << lev-1 << "_feature.xml" ;
		Mat C; //cost matrix
		FileStorage ofs;
		ofs.open(ss.str(), FileStorage::READ);
		if(!ofs.isOpened())
		{
			cout << "ERROR: xml file open failed\n";
			return -1;
		}
		ofs[featCode] >> C;
		if(!C.data)
		{
			return ERR_CLUSTER_FEATURE_NONEXIST;
		}

//		float std = 10; //strongly recommend to set it adaptative to data distribution
		Mat S = Mat::zeros(C.size(), CV_32FC1); //similarity matrix
		for(int row = 0; row < S.rows; row++)
		{
			float sum_row = 0;
			for(int k = 0; k < S.cols; k++)
				sum_row += C.at<float>(row, k);
			float std = sum_row / S.cols;
			for(int col = row+1; col < S.cols; col++)
			{
				S.at<float>(row, col) = exp(pow(C.at<float>(row,col),2)*(-1)/2/pow(std,2));
			}
		}
		for(int row = 0; row < S.rows; row++)
			for(int col = 0; col < row; col++)
				S.at<float>(row, col) = S.at<float>(col, row);

		//diagonal matrix
		Mat D = Mat::zeros(S.size(), CV_32FC1);
		for(int i = 0; i < S.rows; i++)
		{
			float sum_row = 0;
			for(int j = 0; j < S.cols; j++)
			{
				sum_row += S.at<float>(i,j);
			}
			D.at<float>(i, i) = 1 / sqrt(sum_row);
		}

		//graph Laplacian
		Mat L = D * S * D;
		Mat eigenvalues, eigenvectors;
		eigen(L, eigenvalues, eigenvectors);
		eigenvectors = eigenvectors.t();
		eigenvectors.colRange(0, clusterCount-1).copyTo(samples);
		for(int i = 0; i < samples.rows; i++)
			normalize(samples.row(i), samples.row(i));

		//kmeans clustering
		Mat centers, labels;
		kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
		cout << "finish spectral clustering for level: " << lev << endl;

		//restore cluster relationship of current level
		vector<vector<int> > clusterTree(centers.rows);
		for(size_t k = 0; k < (size_t)samples.rows; k++)
		{
			clusterTree[labels.at<int>(k, 0)].push_back(k);
		}
		_levelTree.push_back(clusterTree);


		//train knn with kmeans cluster centers
		CvKNearest knn;
		for(size_t k = 0; k < (size_t)centers.rows; k++)
		{
			Mat center(1, centers.cols, centers.type());
			centers.row(k).copyTo(center);
			Mat label = Mat::ones(1, 1, CV_32F)*k;
			if(k == 0)
			{
				knn.train(center, label, Mat(), false, 1, false);
			}
			else
			{
				knn.train(center, label, Mat(), false, 1, true);
			}

		}

		//find the nearest grasp for each cluster center from current level
		Mat lab, rsp, dis;
		knn.find_nearest(samples, 1, lab, rsp, dis);

		vector<float> minVal(centers.rows, INFINITY);
		vector<int> minIndex(centers.rows, -1);
		for(size_t i = 0; i < (size_t)samples.rows; i++)
		{
			int cluster_id = lab.at<float>(i, 0);
			if(dis.at<float>(i, 0) < minVal[cluster_id] && dis.at<float>(i, 0) >=0)
			{
				minVal[cluster_id] = dis.at<float>(i, 0);
				minIndex[cluster_id] = _minIndex[lev-1][i];
			}
		}

		//restore the nearest neighbor grasp for later visualization
		_minIndex.push_back(minIndex);

		//save average distance between current clusters as new level features
		Mat C_new = Mat::zeros(centers.rows, centers.rows, CV_32FC1);
		for(int i = 0; i < centers.rows; i++)
			for(int j = i+1; j < centers.rows; j++)
			{
				float sum_cost = 0;
				for(int k = 0; k < (int)clusterTree[i].size(); k++)
					for(int l = 0; l < (int)clusterTree[j].size(); l++)
						sum_cost += C.at<float>(clusterTree[i][k], clusterTree[j][l]);
				C_new.at<float>(i, j) = sum_cost / (int)clusterTree[i].size() / (int)clusterTree[j].size();
			}
		for(int i = 0; i < C_new.rows; i++)
			for(int j = 0; j < i; j++)
				C_new.at<float>(i, j) = C_new.at<float>(j, i);

		// write binary image of cluster centers
		ss.str("");
		ss << workDir + "/cluster/spectral/" + featCode + "/level_" << lev << "_feature.xml" ;
		FileStorage fs;
		fs.open(ss.str(), FileStorage::WRITE);
		fs << featCode << C_new;
		fs.release();
	}

	_levelTree.push_back(vector< vector<int> >());
	_levelTree[totalLevel-1].push_back(vector<int>(1,0));
	_levelTree[totalLevel-1].push_back(vector<int>(1,1));
	if(baselevelSize >> (totalLevel-1) >= 3) _levelTree[totalLevel-1].push_back(vector<int>(1,2));

	//save hierarchical tree structure to html
	drawGraspHierarchyHtml(workDir, string("spectral"), featCode, 0, totalLevel-1, false);
	drawGraspHierarchyHtml(workDir, string("spectral"), featCode, 0, totalLevel-2, true);
	return 0;
}

int HiCluster :: cluster(string workDir, vector<string> featureType)
{
	assert(workDir != "" && (size_t)featureType.size() > 0);

	int stat = 0;
	stringstream ss;
	ss << "mkdir " + workDir + "/cluster";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	if("KMEANS" == _clusterType)
	{
		stat = kmeansCluster(workDir, featureType);
		if(stat) return stat;
	}
	else if("SPECTRAL" == _clusterType)
	{
		stat = spectralCluster(workDir, featureType);
		if(stat) return stat;
		stat = flatSpectralCluster(workDir, featureType);
		if(stat) return stat;
	}
	else
	{
		return ERR_CFG_CLUSTER_INVALID;
	}

	return 0;
}

int HiCluster::getHiClusterAllFromGTEA(vector<vector<FeatureInfo> > featureInfo)
{
	Dataset db = dataset_setup("GTEA");
	string database = db.getDatasetName();
	stringstream ss;

	//calculate distance matrix
	int num_sample = (int)featureInfo[0].size();
	Mat distM = Mat::zeros(num_sample, num_sample, CV_32FC1);
	for(int i = 0; i < num_sample; i++)
		for(int j = i+1; j < num_sample; j++)
		{
			//calculate distance of two samples
			distM.at<float>(i,j) = calcDissimilarity(featureInfo[0][i], featureInfo[0][j], _feature_set, _weight_set);
		}
	for(int i = 0; i < num_sample; i++)
		for(int j = 0; j < i; j++)
			distM.at<float>(i,j) = distM.at<float>(j,i);

	//top directory for saving clustering results	
	
	char dirName[500];
	sprintf(dirName, "%s", "all");
	for(int i = 0; i < (int)_feature_set.size(); i++)
		sprintf(dirName, "%s_%s%d", dirName, _feature_set[i].c_str(), (int)(_weight_set[i]*100));
	string fDir(dirName);

	ss.str("");
	ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/";
	cout << ss.str() << endl;
	system(ss.str().c_str());
	
	//hierarchical clustering levels, with 1/2 convergence rate
	int baselevelSize = distM.rows;
	int totalLevel = floor( log2(baselevelSize) );
	Mat prev_label(baselevelSize, 1, CV_32S); //clustering result of lower level
	for(int i = 0; i < prev_label.rows; i++)
		prev_label.at<int>(i,0) = i;
	for(int lev = 1; lev < totalLevel; lev++)
	{
		//spectral clustering
		int k = baselevelSize >> lev;
		Mat labels, centers;
		getSpectralCluster(distM, k, labels, centers);

		//restore cluster relationship of current level
		vector<vector<int> > clusterTree(centers.rows);
		for(int m = 0; m < labels.rows; m++)
		{
			clusterTree[labels.at<int>(m, 0)].push_back(m);
		}

		//save average distance between current clusters as new distance matrix
		Mat C_new = Mat::zeros(centers.rows, centers.rows, CV_32FC1);
		for(int i = 0; i < centers.rows; i++)
			for(int j = i+1; j < centers.rows; j++)
			{
				float sum_cost = 0;
				for(int m = 0; m < (int)clusterTree[i].size(); m++)
					for(int l = 0; l < (int)clusterTree[j].size(); l++)
						sum_cost += distM.at<float>(clusterTree[i][m], clusterTree[j][l]);
				C_new.at<float>(i, j) = sum_cost / (int)clusterTree[i].size() / (int)clusterTree[j].size();
			}
		for(int i = 0; i < C_new.rows; i++)
			for(int j = 0; j < i; j++)
				C_new.at<float>(i, j) = C_new.at<float>(j, i);
		distM = C_new;

		//save result		
		ss.str("");
		ss << "rm -r /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
		cout << ss.str() << endl;
		system(ss.str().c_str());
		
		ss.str("");
		ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
		cout << ss.str() << endl;
		system(ss.str().c_str());

		ss.str("");
		ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/log.txt";
		ofstream outfile(ss.str().c_str(), ios::out);
		vector<int> label_count(k, 0);
		for(int i = 0; i < baselevelSize; i++)
		{
			int label_v = labels.at<int>(prev_label.at<int>(i,0), 0);
			ss.str("");
	        ss << featureInfo[0][i].seqNum << "\t" << featureInfo[0][i].frameNum << "\t";
	        ss << "(" << featureInfo[0][i].box.x << "," << featureInfo[0][i].box.y << "," << featureInfo[0][i].box.width << "," << featureInfo[0][i].box.height << ")\t";
			ss << label_v << endl;
	        outfile << ss.str();

			ss.str("");
			ss << "/home/cai-mj/_GTA/img/" + db.getSequence(featureInfo[0][i].seqNum).seqName+ "/" << setw(8) << setfill('0') << featureInfo[0][i].frameNum << ".jpg";
			Mat img = imread(ss.str());
			img = img(featureInfo[0][i].box).clone();
			ss.str("");
			ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/" << label_v << "_" << label_count[label_v] << ".jpg";
			imwrite(ss.str(), img);

			label_count[label_v]++;
			prev_label.at<int>(i,0) = label_v;
		}
		outfile.close();
	}
	
	return 0;
}

int HiCluster::getClusterAllFromIntel(vector<vector<FeatureInfo> > featureInfo)
{
	Dataset db = dataset_setup("Intel");
	string database = db.getDatasetName();
	stringstream ss;
	
	//calculate distance matrix
	int num_sample = (int)featureInfo[0].size();
	Mat distM = Mat::zeros(num_sample, num_sample, CV_32FC1);
	for(int i = 0; i < num_sample; i++)
		for(int j = i+1; j < num_sample; j++)
		{
			//calculate distance of two samples
			distM.at<float>(i,j) = calcDissimilarity(featureInfo[0][i], featureInfo[0][j], _feature_set, _weight_set);
		}
	for(int i = 0; i < num_sample; i++)
		for(int j = 0; j < i; j++)
			distM.at<float>(i,j) = distM.at<float>(j,i);

	//spectral clustering
	int k = INTEL_OBJECT_SIZE;
	Mat labels, centers;
	getSpectralCluster(distM, k, labels, centers);
	
	//save results
	char dirName[500];
	sprintf(dirName, "%s", "all");
	for(int i = 0; i < (int)_feature_set.size(); i++)
		sprintf(dirName, "%s_%s%d", dirName, _feature_set[i].c_str(), (int)(_weight_set[i]*100));
	string fDir(dirName);
	
	ss.str("");
	ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/";
	cout << ss.str() << endl;
	system(ss.str().c_str());
	
	ss.str("");
	ss << "rm -r /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
	cout << ss.str() << endl;
	system(ss.str().c_str());
	
	ss.str("");
	ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
	cout << ss.str() << endl;
	system(ss.str().c_str());

	ss.str("");
	ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/log.txt";
	ofstream outfile(ss.str().c_str(), ios::out);
	vector<int> label_count(k, 0);
	for(int i = 0; i < labels.rows; i++)
	{
		int label_v = labels.at<int>(i,0);
		ss.str("");
        ss << featureInfo[0][i].seqNum << "\t" << featureInfo[0][i].frameNum << "\t";
        ss << "(" << featureInfo[0][i].box.x << "," << featureInfo[0][i].box.y << "," << featureInfo[0][i].box.width << "," << featureInfo[0][i].box.height << ")\t";
		ss << label_v << endl;
        outfile << ss.str();

		ss.str("");
		ss << "/home/cai-mj/_GTA/img/" + db.getSequence(featureInfo[0][i].seqNum).seqName+ "/" << setw(10) << setfill('0') << featureInfo[0][i].frameNum << ".jpg";
		Mat img = imread(ss.str());
		img = img(featureInfo[0][i].box).clone();
		ss.str("");
		ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/" << label_v << "_" << label_count[label_v] << ".jpg";
		imwrite(ss.str(), img);

		label_count[label_v]++;
	}
	outfile.close();
	return 0;
}

int HiCluster::getClusterFromIntel(vector<vector<FeatureInfo> > featureInfo)
{
	Dataset db = dataset_setup("Intel");
	string database = db.getDatasetName();
	stringstream ss;

	for(int oid = 0; oid < (int)featureInfo.size(); oid++)
	{
		//calculate distance matrix
		int num_sample = (int)featureInfo[oid].size();
		Mat distM = Mat::zeros(num_sample, num_sample, CV_32FC1);
		for(int i = 0; i < num_sample; i++)
	        for(int j = i+1; j < num_sample; j++)
	        {
	        	//calculate distance of two samples
	        	distM.at<float>(i,j) = calcDissimilarity(featureInfo[0][i], featureInfo[0][j], _feature_set, _weight_set);
	        }
	    for(int i = 0; i < num_sample; i++)
	        for(int j = 0; j < i; j++)
	            distM.at<float>(i,j) = distM.at<float>(j,i);

		//try different cluster number K without any prior knowledge of ground truth
		for(int k = 2; k <= 10; k++)
		{
			//spectral clustering
			Mat labels, centers;
			getSpectralCluster(distM, k, labels, centers);
			
			//save results
			string fDir = "object-" + to_string(oid+1);
			for(int i = 0; i < (int)_feature_set.size(); i++)
				fDir = fDir + "_" + _feature_set[i] + to_string((int)(_weight_set[i]*100));
			
			ss.str("");
			ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/";
			cout << ss.str() << endl;
			system(ss.str().c_str());
			
			ss.str("");
			ss << "rm -r /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
			cout << ss.str() << endl;
			system(ss.str().c_str());
			
			ss.str("");
			ss << "mkdir /home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters";
			cout << ss.str() << endl;
			system(ss.str().c_str());

			ss.str("");
			ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/log.txt";
			ofstream outfile(ss.str().c_str(), ios::out);
			vector<int> label_count(k, 0);
			for(int i = 0; i < labels.rows; i++)
			{
				int label_v = labels.at<int>(i,0);
				ss.str("");
	            ss << featureInfo[oid][i].seqNum << "\t" << featureInfo[oid][i].frameNum << "\t";
	            ss << "(" << featureInfo[oid][i].box.x << "," << featureInfo[oid][i].box.y << "," << featureInfo[oid][i].box.width << "," << featureInfo[oid][i].box.height << ")\t";
				ss << label_v << endl;
	            outfile << ss.str();

				ss.str("");
				ss << "/home/cai-mj/_GTA/img/" + db.getSequence(featureInfo[oid][i].seqNum).seqName+ "/" << setw(10) << setfill('0') << featureInfo[oid][i].frameNum << ".jpg";
				Mat img = imread(ss.str());
				img = img(featureInfo[oid][i].box).clone();
				ss.str("");
				ss << "/home/cai-mj/_GTA/grasp/" + database + "/version" << _version << "/" + fDir + "/" << k << "-clusters/" << label_v << "_" << label_count[label_v] << ".jpg";
				imwrite(ss.str(), img);

				label_count[label_v]++;
			}
			outfile.close();
		}
		cout << "finish clustering for objectid " << oid+1 << endl;
	}
	return 0;
}

void HiCluster :: drawGraspHierarchyHtml(string workDir, string clusterCode, string featCode, int startLevel, int endLevel, bool isMultiple)
{
	stringstream ss;
	int levels = endLevel - startLevel + 1;
	if(isMultiple)
	{
		int htmlCount = 0;

		ss.str("");
		ss << workDir + "/cluster/" + clusterCode + "/" + featCode + "/grasp_hierarchy_level[" << startLevel << "-" << endLevel << "].html";
		cout << "Writing: " << ss.str() << endl;
		ofstream fs_grasp(ss.str().c_str());
		if(!fs_grasp.is_open())
		{
			cout << "ERROR: Check filename\n";
			return;
		}

		fs_grasp << "<HTML>\n";
		fs_grasp << "<form name=\"form1\" method=\"post\" action=\"http://www.hci.iis.u-tokyo.ac.jp/~cai-mj/process_kmeans.php\">\n";
		fs_grasp << "<H3>" << featCode << " level range [" << startLevel << " : " << endLevel << "]</H3>\n";

		for(int cluster_id = 0; cluster_id < (int)_levelTree[endLevel].size(); cluster_id++)
			for(int sample_id = 0; sample_id < (int)_levelTree[endLevel][cluster_id].size(); sample_id++)
			{
				//prepare tree structure for visualization
				vector< vector< vector<int> > > subTree(levels);
				subTree[0].push_back(vector<int>(1, _levelTree[endLevel][cluster_id][sample_id]));
				for(int i = 1; i < levels; i++)
				{
					for(int j = 0; j < (int)subTree[i-1].size(); j++)
						for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
						{
							subTree[i].push_back(_levelTree[endLevel-i][subTree[i-1][j][k]]);
						}
				}

				//precompute column spans for visualization
				vector< vector< vector<int> > > subspanTree(levels-1);
				for(int i = levels-1; i > 0; i--)
				{
					int count = 0;
					for(int j = 0; j < (int)subTree[i-1].size(); j++)
					{
						subspanTree[i-1].push_back(vector<int>());
						for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
						{
							if(i == levels-1)
								subspanTree[i-1][j].push_back((int)subTree[i][count].size());
							else
							{
								int sum = 0;
								for(int m = 0; m < (int)subspanTree[i][count].size(); m++)
								{
									sum += subspanTree[i][count][m];
								}
								subspanTree[i-1][j].push_back(sum);
							}
							count++;
						}
					}
				}


				fs_grasp << "<br>\n";
				fs_grasp << "<H5>No:" << htmlCount << "</H5>\n";
				fs_grasp << "<table border = \"1\" >\n";

				//write lowest level
				fs_grasp << "  <tr>\n";
				for(int j = 0; j < (int)subTree[levels-1].size(); j++)
				{
					for(int k = 0; k < (int)subTree[levels-1][j].size(); k++)
					{
						fs_grasp << "    <td>\n";
/*						ss.str("");
						ss << "<img src=\"../../../source/img/";
						ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_img.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
						fs_grasp << ss.str();
						fs_grasp << "    <br>\n";
						ss.str("");
						ss << "<img src=\"../../../source/hand/";
						ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
						fs_grasp << ss.str();
						fs_grasp << "    <br>\n";
*/						ss.str("");
						ss << "<img src=\"../../../source/hand/";
						ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
						fs_grasp << ss.str();
						fs_grasp << "    </td>\n";
					}
				}
				fs_grasp << "  </tr>\n";

				//write upper levels
				for(int i = levels-1; i > 0; i--)
				{
					fs_grasp << "  <tr>\n";
					for(int j = 0; j < (int)subTree[i-1].size(); j++)
					{
						for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
						{
							int colSpan = subspanTree[i-1][j][k];
							ss.str("");
							ss << "    <td colspan = \"" << colSpan << "\" align = \"center\">\n";
							fs_grasp << ss.str();

							ss.str("");
							ss << "<img src=\"../../../source/hand/";
							ss << setw(8) << setfill('0') << _minIndex[endLevel+1-i][subTree[i-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
							fs_grasp << ss.str();
							fs_grasp << "    <br>\n";
							ss.str("");
							ss << "<input type=\"checkbox\" name=\"level" << endLevel+1-i << "[]\" value=\"g\"><font size=-1>good</font>";
//							fs_grasp << ss.str();
							ss.str("");
							ss << "<input type=\"checkbox\" name=\"level" << endLevel+1-i << "[]\" value=\"b\"><font size=-1>bad</font>\n";
//							fs_grasp << ss.str();
							fs_grasp << "    </td>\n";
						}

					}

					fs_grasp << "  </tr>\n";
				}

				fs_grasp << "</table>\n";
				htmlCount++;


			}

		fs_grasp << "<br>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<input type=\"submit\" name=\"submit\" value=\"SUBMIT\">\n";
		fs_grasp << "</form>\n";
		fs_grasp << "</HTML>\n";
	}
	else
	{
		//prepare tree structure for visualization
		vector< vector< vector<int> > > subTree(levels);
		for(int cluster_id = 0; cluster_id < (int)_levelTree[endLevel].size(); cluster_id++)
			for(int sample_id = 0; sample_id < (int)_levelTree[endLevel][cluster_id].size(); sample_id++)
			{
				subTree[0].push_back(vector<int>(1, _levelTree[endLevel][cluster_id][sample_id]));
			}

		for(int i = 1; i < levels; i++)
			for(int j = 0; j < (int)subTree[i-1].size(); j++)
				for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
				{
					subTree[i].push_back(_levelTree[endLevel-i][subTree[i-1][j][k]]);
				}

		//precompute column spans for visualization
		vector< vector< vector<int> > > subspanTree(levels-1);
		for(int i = levels-1; i > 0; i--)
		{
			int count = 0;
			for(int j = 0; j < (int)subTree[i-1].size(); j++)
			{
				subspanTree[i-1].push_back(vector<int>());
				for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
				{
					if(i == levels-1)
						subspanTree[i-1][j].push_back((int)subTree[i][count].size());
					else
					{
						int sum = 0;
						for(int m = 0; m < (int)subspanTree[i][count].size(); m++)
						{
							sum += subspanTree[i][count][m];
						}
						subspanTree[i-1][j].push_back(sum);
					}
					count++;
				}
			}
		}

		ss.str("");
		ss << workDir + "/cluster/" + clusterCode + "/" + featCode + "/grasp_hierarchy_level[" << startLevel << "-" << endLevel << "].html";
		cout << "Writing: " << ss.str() << endl;

		ofstream fs_grasp(ss.str().c_str());
		if(!fs_grasp.is_open())
		{
			cout << "ERROR open html file\n";
			return;
		}

		fs_grasp << "<HTML>\n";
		fs_grasp << "<form name=\"form1\" method=\"post\" action=\"http://www.hci.iis.u-tokyo.ac.jp/~cai-mj/process_kmeans.php\">\n";
		fs_grasp << "<H3>" << featCode << " level range [" << startLevel << " : " << endLevel << "]</H3>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<table border = \"1\" >\n";

		//write lowest level
		fs_grasp << "  <tr>\n";
		for(int j = 0; j < (int)subTree[levels-1].size(); j++)
		{
			for(int k = 0; k < (int)subTree[levels-1][j].size(); k++)
			{
				fs_grasp << "    <td>\n";
/*				ss.str("");
				ss << "<img src=\"../../../source/img/";
				ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_img.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
				fs_grasp << ss.str();
				fs_grasp << "    <br>\n";
				ss.str("");
				ss << "<img src=\"../../../source/hand/";
				ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
				fs_grasp << ss.str();
				fs_grasp << "    <br>\n";
*/				ss.str("");
				ss << "<img src=\"../../../source/hand/";
				ss << setw(8) << setfill('0') << _minIndex[startLevel][subTree[levels-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
				fs_grasp << ss.str();
				fs_grasp << "    </td>\n";
			}
		}
		fs_grasp << "  </tr>\n";

		//write upper levels
		for(int i = levels-1; i > 0; i--)
		{
			fs_grasp << "  <tr>\n";
			for(int j = 0; j < (int)subTree[i-1].size(); j++)
			{
				for(int k = 0; k < (int)subTree[i-1][j].size(); k++)
				{
					int colSpan = subspanTree[i-1][j][k];
					ss.str("");
					ss << "    <td colspan = \"" << colSpan << "\" align = \"center\">\n";
					fs_grasp << ss.str();

					ss.str("");
					ss << "<img src=\"../../../source/hand/";
					ss << setw(8) << setfill('0') << _minIndex[endLevel+1-i][subTree[i-1][j][k]] << "_handroi.jpg" << "\"  WIDTH=\"120\" BORDER=\"0\" >\n";
					fs_grasp << ss.str();
					fs_grasp << "    <br>\n";
					ss.str("");
					ss << "<input type=\"checkbox\" name=\"level" << endLevel+1-i << "[]\" value=\"g\"><font size=-1>good</font>";
//					fs_grasp << ss.str();
					ss.str("");
					ss << "<input type=\"checkbox\" name=\"level" << endLevel+1-i << "[]\" value=\"b\"><font size=-1>bad</font>\n";
//					fs_grasp << ss.str();
					fs_grasp << "    </td>\n";
				}

			}

			fs_grasp << "  </tr>\n";
		}

		fs_grasp << "</table>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<br>\n";
		fs_grasp << "<input type=\"submit\" name=\"submit\" value=\"SUBMIT\">\n";
		fs_grasp << "</form>\n";
		fs_grasp << "</HTML>\n";

	}

	return;
}
