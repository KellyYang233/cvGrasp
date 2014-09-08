/*
 *  cvGrasp.cpp
 *  Discovering grasp types using computer vision
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "cvGrasp.hpp"

Size arrange_pos_neg(Mat &trainData, Mat &labels)
{	
	Mat posData, negData;
	int sum_pos=0, sum_neg=0;
	for(int i = 0; i < (int)labels.rows; i++)
	{
		if(labels.at<float>(i,0) > 0)
		{
			sum_pos++;
			Mat temp;
			trainData.row(i).copyTo(temp);
			posData.push_back(temp);
		}
		else
		{
			sum_neg++;
			Mat temp;
			trainData.row(i).copyTo(temp);
			negData.push_back(temp);
		}
	}

	trainData = posData;
	trainData.push_back(negData);
	labels = Mat::ones(posData.rows, 1, CV_32F);
	Mat label = Mat::ones(negData.rows, 1, CV_32F) * 0;
	labels.push_back(label);

	cout << "positive samples: " << sum_pos << " negative samples: " << sum_neg << endl;
	return Size(sum_pos, sum_neg);
}

void balance_pos_neg(Mat &trainData, Mat &labels, int sum_pos, int sum_neg, int class_num)
{
	int num_resample = (sum_neg+sum_pos)/class_num - sum_pos; 
	int num_resample_high = (sum_neg+sum_pos)*2/class_num - sum_pos;
	if(num_resample > 0)
	{
		Mat posData, negData;
		trainData.rowRange(0,sum_pos).copyTo(posData);
		trainData.rowRange(sum_pos, sum_pos+sum_neg).copyTo(negData);
		
		for(int j = 0; j < num_resample; j++)
		{
			int r = rand()%sum_pos;
			Mat temp;
			posData.row(r).copyTo(temp);
			posData.push_back(temp);
		}
		trainData = posData;
		trainData.push_back(negData);
		labels = Mat::ones(posData.rows, 1, CV_32F);
		Mat label = Mat::ones(negData.rows, 1, CV_32F) * 0;
		labels.push_back(label);
		cout << "resampling " << num_resample << " positive samples\n";
	}	
	else if(num_resample_high < 0)
	{
		num_resample = (sum_neg+sum_pos)*2/class_num;
		Mat posData, negData;
		trainData.rowRange(0,sum_pos).copyTo(posData);
		trainData.rowRange(sum_pos, sum_pos+sum_neg).copyTo(negData);
		
		vector<int> ind = randsample(posData.rows, num_resample);
		posData = indMatrixRows<float>(posData, ind);
		trainData = posData;
		trainData.push_back(negData);
		labels = Mat::ones(posData.rows, 1, CV_32F);
		Mat label = Mat::ones(negData.rows, 1, CV_32F) * 0;
		labels.push_back(label);
		cout << "downsampling to " << num_resample << " positive samples\n";
	}

	return;
}

float getAccuracy(Mat &confMatrix, vector<Grasp> &freqGrasps)
{	
	float sum = 0;
	
	//first make the sum of each row equal to 1, which makes recall at diagonal element
	Mat rowWeight = Mat::zeros(confMatrix.rows, 1, CV_32F);
	Mat rMatrix(confMatrix.size(), CV_32F);
	confMatrix.copyTo(rMatrix);
    for(int i = 0; i < confMatrix.rows; i++)
    {
        float rowSum = 0;
        for(int j = 0; j < confMatrix.cols; j++)
            rowSum += confMatrix.at<float>(i,j);
		rowWeight.at<float>(i,0) = rowSum;
		sum += rowSum;
        rowSum = rowSum + 0.000001;
        for(int j = 0; j < confMatrix.cols; j++)
            rMatrix.at<float>(i,j) = confMatrix.at<float>(i,j)/rowSum;
    }
	rowWeight = rowWeight/sum;

	//compute accuracy
	float accuracy = 0.0;
	cout << "recall for each class: \n";
	for(int i = 0; i < rMatrix.rows; i++)
	{
		accuracy += rMatrix.at<float>(i,i) * rowWeight.at<float>(i,0);
		cout << i << " " << freqGrasps[i].graspType << "\t";
		cout << " : " << rMatrix.at<float>(i,i) << " * " << rowWeight.at<float>(i,0) << " = ";
		cout << rMatrix.at<float>(i,i) * rowWeight.at<float>(i,0) << endl;
	}
	
	return accuracy;	
}

float getF1(Mat &confMatrix, Mat &confMatrix_r, vector<Grasp> &freqGrasps)
{
	float sum = 0; //total number of samples
	
	//first make the sum of each row equal to 1, which makes recall at diagonal element	
	Mat rowWeight = Mat::zeros(confMatrix.rows, 1, CV_32F);
	Mat rMatrix(confMatrix.size(), CV_32F);
	confMatrix.copyTo(rMatrix);
    for(int r = 0; r < confMatrix.rows; r++)
    {
        float rowSum = 0;
        for(int c = 0; c < confMatrix.cols; c++)
            rowSum += confMatrix.at<float>(r,c);
		rowWeight.at<float>(r,0) = rowSum;
		sum += rowSum;
        rowSum = rowSum + 0.000001;
        for(int c = 0; c < confMatrix.cols; c++)
            rMatrix.at<float>(r,c) = confMatrix.at<float>(r,c)/rowSum;
    }
	rowWeight = rowWeight/sum;

	//second make the sum of each column equal to 1, which makes precision at diagonal element
	Mat colWeight = Mat::zeros(confMatrix.rows, 1, CV_32F);
	Mat pMatrix(confMatrix.size(), CV_32F);
	confMatrix.copyTo(pMatrix);
    for(int c = 0; c < confMatrix.cols; c++)
    {
        float colSum = 0;
        for(int r = 0; r < confMatrix.rows; r++)
            colSum += confMatrix.at<float>(r,c);
		colWeight.at<float>(c,0) = colSum;
        colSum = colSum + 0.000001;
        for(int r = 0; r < confMatrix.rows; r++)
            pMatrix.at<float>(r,c) = confMatrix.at<float>(r,c)/colSum;
    }
	colWeight = colWeight/sum;

	//compute average F1
	float F1 = 0;
	cout << "F1 score for each class: \n";
	for(int r = 0; r < confMatrix.rows; r++)
	{
		float f1 = 2*pMatrix.at<float>(r,r)*rMatrix.at<float>(r,r) / (pMatrix.at<float>(r,r)+rMatrix.at<float>(r,r)+0.000001);
		F1 += f1*rowWeight.at<float>(r,0);

		cout << r << " " << freqGrasps[r].graspType << "\t";
		cout << " : " << f1 << "(" << pMatrix.at<float>(r,r) << "," << rMatrix.at<float>(r,r) << ") * " << rowWeight.at<float>(r,0) << " = ";
		cout << f1 * rowWeight.at<float>(r,0) << endl;
	}

	rMatrix.copyTo(confMatrix_r);
	return F1;
}

Mat getCorrelation(Mat &confMatrix)
{	
	
	//first make the sum of each row equal to 1, which makes recall at diagonal element
	Mat rowWeight = Mat::zeros(confMatrix.rows, 1, CV_32F);
	Mat rMatrix(confMatrix.size(), CV_32F);
	confMatrix.copyTo(rMatrix);
    for(int i = 0; i < confMatrix.rows; i++)
    {
        float rowSum = 0;
        for(int j = 0; j < confMatrix.cols; j++)
            rowSum += confMatrix.at<float>(i,j);
		rowWeight.at<float>(i,0) = rowSum;
        rowSum = rowSum + 0.000001;
        for(int j = 0; j < confMatrix.cols; j++)
            rMatrix.at<float>(i,j) = confMatrix.at<float>(i,j)/rowSum;
    }

	Mat correlation = Mat::zeros(confMatrix.size(), CV_32F);
	for(int i = 0; i < correlation.rows; i++)
		for(int j = i+1; j < correlation.cols; j++)
			correlation.at<float>(i,j) = (confMatrix.at<float>(i,j)+confMatrix.at<float>(j,i))/(rowWeight.at<float>(i,0)+rowWeight.at<float>(j,0));

	for(int i = 0; i < correlation.rows; i++)
		for(int j = 0; j < i; j++)
			correlation.at<float>(i,j) = correlation.at<float>(j,i);

	return correlation;
}

void recordFP(Dataset &db, int version, Mat &label, Mat &classification, vector<GraspNode> &nodes, vector<Grasp> &freqGrasps)
{
	string rootDir = db.getDataDir();
	string dbName = db.getDatasetName();
	int n = classification.rows;
	if(n != label.rows || n != (int)nodes.size())
	{
		cout << "[recordFP] ERROR: unequal rows for label-" << label.rows << ", classfication-" << n << " and nodes-" << (int)nodes.size() << endl;
		exit(-1);
	}
    for(int i = 0; i < n; i++)
    {
        int lgt = label.at<int>(i,0);
		if(freqGrasps[lgt].graspType != nodes[i].graspType)
		{
				cout << "[recordFP] WARNING: unmatched grasp type label: " << freqGrasps[lgt].graspType << " node: " << nodes[i].graspType << " " << nodes[i].seqName.c_str() << " " << nodes[i].frameid << endl;
		}
        int lc, dummy;
        matrixMaxValue<float>(classification.rowRange(i,i+1), dummy, lc);
        if(lc != lgt)
        {
        	//color image
        	char filename[500];
			filename[0] = 0;
			sprintf(filename, "%s/img/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			Mat img = imread(filename);
			if(!img.data)
			{
				cout << "[recordFP] ERROR: failed to read image " << string(filename) << endl;
			}
			Mat imgroi = img(nodes[i].roi).clone();
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/FP/%s_%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, imgroi);

			//hand probability
			filename[0] = 0;
			sprintf(filename, "%s/hand/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			Mat hand = imread(filename);
			if(!hand.data)
			{
				cout << "[recordFP] ERROR: failed to read image " << string(filename) << endl;
			}
			Mat handroi = hand(nodes[i].roi).clone();			
			
			Mat p_hand(handroi.size(), CV_32FC1);
			Mat bgr[3];
			split(handroi, bgr);
			for(int r = 0; r < p_hand.rows; r++)
		    {
		        for(int c = 0; c < p_hand.cols; c++)
		        {
		            p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
		        }
		    }
			int idx;
			vector<vector<Point> > co;
			Mat mask;
			getContourBig(p_hand, mask, 0.3, co, idx);
			handroi *= 0;
			imgroi.copyTo(handroi, mask);
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/FP/%s_%s_%d_p.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, handroi);

			//sift visualization
//			Rect roi_obj = getBox_object(nodes[i].roi, hand);
//			imgroi = img(roi_obj).clone();
			visualizeSIFT(imgroi);
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/FP/%s_%s_%d_sift.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, imgroi);
        }
		else
		{
			char filename[500];
			filename[0] = 0;
			sprintf(filename, "%s/img/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			Mat img = imread(filename);
			if(!img.data)
			{
				cout << "[recordFP] ERROR: failed to read image " << string(filename) << endl;
			}
			Mat imgroi = img(nodes[i].roi).clone();
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/TP/%s_%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, imgroi);

			//hand probability
			filename[0] = 0;
			sprintf(filename, "%s/hand/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			Mat hand = imread(filename);
			if(!hand.data)
			{
				cout << "[recordFP] ERROR: failed to read image " << string(filename) << endl;
			}
			Mat handroi = hand(nodes[i].roi).clone();	
			
			Mat p_hand(handroi.size(), CV_32FC1);
			Mat bgr[3];
			split(handroi, bgr);
			for(int r = 0; r < p_hand.rows; r++)
		    {
		        for(int c = 0; c < p_hand.cols; c++)
		        {
		            p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
		        }
		    }
			int idx;
			vector<vector<Point> > co;
			Mat mask;
			getContourBig(p_hand, mask, 0.3, co, idx);
			handroi *= 0;
			imgroi.copyTo(handroi, mask);			
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/TP/%s_%s_%d_p.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, handroi);

			//sift visualization
//			Rect roi_obj = getBox_object(nodes[i].roi, hand);
//			imgroi = img(roi_obj).clone();
			visualizeSIFT(imgroi);
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/TP/%s_%s_%d_sift.jpg", rootDir.c_str(), dbName.c_str(), version, freqGrasps[lc].graspType.c_str(), freqGrasps[lgt].graspType.c_str(), nodes[i].seqName.c_str(), nodes[i].frameid);
			imwrite(filename, imgroi);
		}
    }
}

void writeFP2Html(Dataset &db, int version, vector<Grasp> &freqGrasps)
{
	string dataDir = db.getDataDir();
	string dbName = db.getDatasetName();
	
	char filename[500];
	filename[0] = 0;
	sprintf(filename, "%s/grasp/%s/version%d/temp/falsePositive.html", dataDir.c_str(), dbName.c_str(), version);

	ofstream fs(filename);
	if(!fs.is_open())
	{
		cout << "[writeFP2Html] Error: failed to open file\n";
		return;
	}

	fs << "<HTML>\n";
    fs << "<form name=\"form1\" method=\"post\" action=\"http://rus.hci.iis.u-tokyo.ac.jp:800/~cai-mj/process_kmeans.php\">\n";
    fs << "<H3> False positive and true positive examples of grasp classification</H3>\n";
    fs << "<br>\n";

	for(int i = 0; i < (int)freqGrasps.size(); i++)
	{
		string graspType = freqGrasps[i].graspType;
		fs << "<H4> " + graspType + "</H4>\n";
		filename[0] = 0;
		sprintf(filename, "<img src=\"zz/%s.jpg\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", graspType.c_str());
		fs << filename;
		
		// draw true positive
		fs << "    <H5> True Positive</H5>\n";
		fs << "    <table border = \"1\" >\n";
		
		char cmd[500];
		cmd[0] = 0;
		sprintf(cmd, "ls %s/grasp/%s/version%d/temp/%s/TP/*_p.jpg > TP.txt", dataDir.c_str(), dbName.c_str(), version, graspType.c_str());
		system(cmd);

		ifstream ifs("TP.txt", ios::in);
		string pHandName;
		string imgName;
		string siftName;
		int count = 0;
	    while(ifs >> pHandName)
	    {
	    	if(count%10 == 0)
				fs << "        <tr>\n";

			fs << "            <td>\n";

			pHandName = pHandName.substr(pHandName.find("temp/")+strlen("temp/"));
			imgName = pHandName;
			imgName.erase(pHandName.rfind("_p"), strlen("_p"));
			siftName = pHandName;
			siftName.replace(siftName.rfind("_p"), strlen("_p"), string("_sift"));
			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", imgName.c_str());
			fs << filename;

			fs << "            <br>\n";

			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", pHandName.c_str());
			fs << filename;

			fs << "            <br>\n";

			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", siftName.c_str());
			fs << filename;

			fs << "            <br>\n";

			imgName.erase(0, imgName.find("TP/")+strlen("TP/"));
			imgName.erase(imgName.rfind(".jpg"));
			fs << "            <p align=\"center\"><small>";
			size_t pos = imgName.substr(0, imgName.find_last_of("_")).find_last_of("_");
			fs << imgName.substr(0, pos);
			fs << "            </small></p>";
			fs << "            <p align=\"center\"><small>";
			fs << imgName.substr(pos+1);			
			fs << "            </small></p>";
			
			fs << "            </td>\n";

			count++;
			if(count%10 == 0)
				fs << "        </tr>\n";
	    }
		
		if(count%10 != 0)
			fs << "        </tr>\n";
		fs << "    </table>\n";

		// draw false positive
		fs << "    <H5> False Positive</H5>\n";
		fs << "    <table border = \"1\" >\n";
		ifs.close();
		
		cmd[0] = 0;
		sprintf(cmd, "ls %s/grasp/%s/version%d/temp/%s/FP/*_p.jpg > FP.txt", dataDir.c_str(), dbName.c_str(), version, graspType.c_str());
		system(cmd);

		ifs.open("FP.txt");
		count = 0;
	    while(ifs >> pHandName)
	    {
	    	if(count%10 == 0)
				fs << "        <tr>\n";

			fs << "            <td>\n";

			pHandName = pHandName.substr(pHandName.find("temp/")+strlen("temp/"));
			imgName = pHandName;
			imgName.erase(pHandName.rfind("_p"), strlen("_p"));
			siftName = pHandName;
			siftName.replace(siftName.rfind("_p"), strlen("_p"), string("_sift"));
			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", imgName.c_str());
			fs << filename;

			fs << "            <br>\n";

			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", pHandName.c_str());
			fs << filename;

			fs << "            <br>\n";

			filename[0] = 0;
			sprintf(filename, "            <img src=\"%s\" WIDTH=\"100\" BORDER=\"0\" align=\"center\">\n", siftName.c_str());
			fs << filename;

			fs << "            <br>\n";

			imgName.erase(0, imgName.find("FP/")+strlen("FP/"));
			imgName.erase(imgName.rfind(".jpg"));
			fs << "            <p align=\"center\"><small>";
			size_t pos = imgName.substr(0, imgName.find_last_of("_")).find_last_of("_");
			fs << imgName.substr(0, pos);
			fs << "            </small></p>";
			fs << "            <p align=\"center\"><small>";
			fs << imgName.substr(pos+1);			
			fs << "            </small></p>";
			
			fs << "            </td>\n";

			count++;
			if(count%10 == 0)
				fs << "        </tr>\n";
	    }
		
		if(count%10 != 0)
			fs << "        </tr>\n";
		fs << "    </table>\n";
		fs << "    <br>\n";
	}
	
	fs << "</form>\n";
	fs << "</HTML>\n";
}

void downSampleTrainData(Mat &trainData, Mat &labels, vector<GraspNode> &nodes, int fold_num, int fold_idx)
{
	// parameter checking
	if(fold_num < 2 || fold_idx < 1 || fold_idx > fold_num)
	{
		cout << "[downSampleTrainData]Error: invalid parameter!\n";
		exit(-1);
	}

	// separation of training/test data index
	vector<int> posIdx(0);
	vector<int> negIdx(0);
	for(int i = 0; i < labels.rows; i++)
	{
		if(labels.at<int>(i,0) > 0)
			posIdx.push_back(i);
		else
			negIdx.push_back(i);
	}
	int posNum = (int)posIdx.size();
	int negNum = (int)negIdx.size();
	
	// n fold division of positive samples
	posIdx.erase(posIdx.begin()+(int)(posNum*(fold_idx-1)*1.0/fold_num), posIdx.begin()+(int)(posNum*fold_idx*1.0/fold_num));
cout << "training samples before division: " << posNum << endl;
cout << "training samples after division: " << (int)posIdx.size() << endl;
	// n fold division for each negative grasp type
	map<string, vector<int> > type2idx;
	for(int i = 0; i < (int)negIdx.size(); i++)
	{
		string type = nodes[negIdx[i]].graspType;
        if(type2idx.find(type) != type2idx.end())
            type2idx[type].push_back(negIdx[i]);
        else
            type2idx[type] = vector<int>(0);
	}
	negIdx.clear();
	for(map<string, vector<int> >::iterator it = type2idx.begin(); it != type2idx.end(); it++)
	{
		int num = (int)it->second.size();		
		it->second.erase(it->second.begin()+(int)(num*(fold_idx-1)*1.0/fold_num), it->second.begin()+(int)(num*fold_idx*1.0/fold_num));

		vector<int> temp(it->second);
		for(int j = 0; j < (int)temp.size(); j++)
			negIdx.push_back(temp[j]);
	}
	cout << "negative samples before division: " << negNum << endl;
	cout << "negative samples after division: " << (int)negIdx.size() << endl;

	vector<int> ind(posIdx);
	for(int i = 0; i < (int)negIdx.size(); i++)
		ind.push_back(negIdx[i]);
	trainData = indMatrixRows<float>(trainData, ind);
	labels = indMatrixRows<int>(labels, ind);
	nodes = indVector<GraspNode>(nodes, ind);

}

void downSampleTestData(Mat &data, Mat &labels, vector<GraspNode> &nodes, int fold_num, int fold_idx, bool isRemain)
{
	// parameter checking
	if(fold_num < 2 || fold_idx < 1 || fold_idx > fold_num)
	{
		cout << "[downSampleTestData]Error: invalid parameter!\n";
		exit(-1);
	}
	int n = data.rows;
	if(n != labels.rows || n != (int)nodes.size())
	{
		cout << "[downSampleTestData] ERROR: unequal rows for label-" << labels.rows << ", data-" << n << " and nodes-" << (int)nodes.size() << endl;
		exit(-1);
	}

	map<int, vector<int> > label2idx;
	for(int i = 0; i < labels.rows; i++)
	{
		int label = labels.at<int>(i,0);
        if(label2idx.find(label) != label2idx.end())
            label2idx[label].push_back(i);
        else
            label2idx[label] = vector<int>(0);
	}

	vector<int> ind(0);
	for(map<int, vector<int> >::iterator it = label2idx.begin(); it != label2idx.end(); it++)
	{
		int num = (int)it->second.size();
		if(isRemain)
		{
			// 5 fold division of each grasp type		
			it->second.assign(it->second.begin()+(int)(num*(fold_idx-1)*1.0/fold_num), it->second.begin()+(int)(num*fold_idx*1.0/fold_num));
		}
		else
		{
			// 5 fold division of each grasp type			
			it->second.erase(it->second.begin()+(int)(num*(fold_idx-1)*1.0/fold_num), it->second.begin()+(int)(num*fold_idx*1.0/fold_num));
		}
cout << "type " << it->first << " before divisioin: " << num << endl;
cout << "type " << it->first << " after divisioin: " << (int)it->second.size() << endl;

		vector<int> temp(it->second);
		for(int j = 0; j < (int)temp.size(); j++)
			ind.push_back(temp[j]);
	}

	data = indMatrixRows<float>(data, ind);
	labels = indMatrixRows<int>(labels, ind);
	nodes = indVector<GraspNode>(nodes, ind);
}

void filterTrainData(Dataset &db, int version, Mat &data, Mat &labels, vector<GraspNode> &nodes, bool isFilter)
{
	char filename[300];
	string rootDir = db.getDataDir();
	string dbName = db.getDatasetName();	
	
	for(int i = 0; i < (int)labels.rows; i++)
	{
		if(labels.at<float>(i,0) > 0)
		{
			string graspType = nodes[i].graspType;
			string seqName = nodes[i].seqName;
			int frameid = nodes[i].frameid;			
			if(isFilter)
			{
				// if need check, check the existence of images and update labels
				filename[0] = 0;
				sprintf(filename, "%s/grasp/%s/version%d/temp/%s/filter/%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, graspType.c_str(), seqName.c_str(), frameid);
				Mat img = imread(filename);
				if(!img.data)
					labels.at<float>(i,0) = 0;
			}
			else
			{
				// if no need check, restore images of positive samples
				filename[0] = 0;
				sprintf(filename, "%s/img/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), seqName.c_str(), frameid);
				Mat img = imread(filename);
				if(!img.data)
				{
					cout << "[filterTestData] Error: failed to read image " << filename << endl;
					continue;
				}

				Mat imgroi = img(nodes[i].roi).clone();
				filename[0] = 0;
				sprintf(filename, "%s/grasp/%s/version%d/temp/%s/train/%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, graspType.c_str(), seqName.c_str(), frameid);
				imwrite(filename, imgroi);
			}	
		}
	}

}

void filterTestData(Dataset &db, int version, Mat &data, Mat &labels, vector<GraspNode> &nodes, bool isFilter)
{
	char filename[300];
	string rootDir = db.getDataDir();
	string dbName = db.getDatasetName();	
	vector<int> ind(0);
	
	for(int i = 0; i < (int)labels.rows; i++)
	{		
		string graspType = nodes[i].graspType;
		string seqName = nodes[i].seqName;
		int frameid = nodes[i].frameid;			
		if(isFilter)
		{
			// if need check, check the existence of images and update labels
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/filter/%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, graspType.c_str(), seqName.c_str(), frameid);
			Mat img = imread(filename);
			if(img.data)
				ind.push_back(i);
		}
		else
		{
			// if no need check, restore images of positive samples
			filename[0] = 0;
			sprintf(filename, "%s/img/%s/%s/%08d.jpg", rootDir.c_str(), dbName.c_str(), seqName.c_str(), frameid);
			Mat img = imread(filename);
			if(!img.data)
			{
				cout << "[filterTestData] Error: failed to read image " << filename << endl;
				continue;
			}

			Mat imgroi = img(nodes[i].roi).clone();
			filename[0] = 0;
			sprintf(filename, "%s/grasp/%s/version%d/temp/%s/test/%s_%d.jpg", rootDir.c_str(), dbName.c_str(), version, graspType.c_str(), seqName.c_str(), frameid);
			imwrite(filename, imgroi);
		}		
	}

	if(isFilter)
	{
		data = indMatrixRows<float>(data, ind);
		labels = indMatrixRows<int>(labels, ind);
		nodes = indVector<GraspNode>(nodes, ind);
	}
}

Rect getBox_unfixed(RotatedRect rRect, Mat &img)
{
	Rect roi = rRect.boundingRect();				
				
	//remove arm part: select 1.5*axis_short as effective length of principle axis
	float angle = rRect.angle;
	float axisL = rRect.size.height;
	float axisS = rRect.size.width;
	float scale = axisS*1.5/axisL;
	if(scale > 1) scale = 1;
	if(angle < 90)
	{
		int width = scale*axisL*sin(angle*PI/180) + axisS*cos(angle*PI/180);
		int height = scale*axisL*cos(angle*PI/180) + axisS*sin(angle*PI/180);
		roi = Rect(roi.x+roi.width-width, roi.y, width, height);
	}
	else
	{
		int width = scale*axisL*sin((180-angle)*PI/180) + axisS*cos((180-angle)*PI/180);
		int height = scale*axisL*cos((180-angle)*PI/180) + axisS*sin((180-angle)*PI/180);
		roi = Rect(roi.x, roi.y, width, height);
	}	

	roi.width = min(roi.width, img.cols/2);
	roi.height = min(roi.height, img.rows/2);
	if(roi.x < 0) roi.x = 0;
	if(roi.y < 0) roi.y = 0;
	if(roi.x + roi.width > img.cols) roi.x = img.cols - roi.width;
	if(roi.y + roi.height > img.rows) roi.y = img.rows - roi.height;

	return roi;
}

Rect getBox_fixed(RotatedRect rRect, Mat &img)
{
	int stat = 0;
	Size box_size(320, 160);
	Rect roi = rRect.boundingRect();				
				
	//remove arm part: select 1.5*axis_short as effective length of principle axis
	float angle = rRect.angle;
	float axisL = rRect.size.height;
	float axisS = rRect.size.width;
	float scale = axisS*1.5/axisL;
	if(scale > 1) scale = 1;
	if(angle < 90)
	{
		int width = scale*axisL*sin(angle*PI/180) + axisS*cos(angle*PI/180);
		int height = scale*axisL*cos(angle*PI/180) + axisS*sin(angle*PI/180);
		roi = Rect(roi.x+roi.width-width, roi.y, width, height);
	}
	else
	{
		int width = scale*axisL*sin((180-angle)*PI/180) + axisS*cos((180-angle)*PI/180);
		int height = scale*axisL*cos((180-angle)*PI/180) + axisS*sin((180-angle)*PI/180);
		roi = Rect(roi.x, roi.y, width, height);
	}	

	roi.width = min(roi.width, img.cols/2);
	roi.height = min(roi.height, img.rows/2);
	if(roi.x < 0) roi.x = 0;
	if(roi.y < 0) roi.y = 0;
	if(roi.x + roi.width > img.cols) roi.x = img.cols - roi.width;
	if(roi.y + roi.height > img.rows) roi.y = img.rows - roi.height;

	Mat imgroi = img(roi).clone();
	Mat bgr[3];
    Mat p_hand(imgroi.size(), CV_32FC1);
    split(imgroi, bgr);
    for(int r = 0; r < p_hand.rows; r++)
    {
        for(int c = 0; c < p_hand.cols; c++)
        {
            p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
        }
    }
	Mat mask;
	vector<vector<Point> > co;
	int idx;
	stat = getContourBig(p_hand, mask, 0.3, co, idx);
	Rect r;
	if(stat)
	{
		r = roi;
	}
	else
	{
		r = boundingRect(co[idx]);
	}

	// compute fixed bounding box	
	Point pt;
	pt.x = roi.x + r.x + r.width*0.5;		// center x
	pt.y = roi.y + r.y;						// top y
	
	roi.x = pt.x - box_size.width*0.5;
	roi.y = pt.y;
	roi.width = box_size.width;
	roi.height = box_size.height;
	
	if(roi.x<0) roi.x = 0;
	if(roi.y<0) roi.y = 0;
	if(roi.x+roi.width >=img.cols) roi.x = img.cols - roi.width;
	if(roi.y+roi.height>=img.rows) roi.y = img.rows - roi.height;

	return roi;
}

Rect getBox_object(Rect handroi, Mat &img)
{
	Size box_size(128,128);
/*	int stat = 0;
	Mat imgroi = img(handroi).clone();
	Mat bgr[3];
    Mat p_hand(imgroi.size(), CV_32FC1);
    split(imgroi, bgr);
    for(int r = 0; r < p_hand.rows; r++)
    {
        for(int c = 0; c < p_hand.cols; c++)
        {
            p_hand.at<float>(r, c) = bgr[0].at<uchar>(r, c)/255.0;
        }
    }

	Mat mask;
	float thres = 0.3;
	vector<vector<Point> > co;
	int idx;
	stat = getContourBig(p_hand, mask, thres, co, idx);
	if(stat)
	{
		cout << "[getBox_object] WARNING: no hand region\n";
		return handroi;
	}

	Mat dt;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	distanceTransform(mask, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	minMaxLoc(dt, &minVal, &maxVal, NULL, &maxLoc);

	// select biggest area after thresholding distance transform map
	thres = maxVal/2;
	stat = getContourBig(dt, mask, thres, co, idx);
	if(stat)
	{
		cout << "[getBox_object] WARNING: no hand region after thresholding\n";
		return handroi;
	}

	// calculate center of gravity as palm center
	Moments moment = moments(mask, true);
	Point palm_center(moment.m10/moment.m00, moment.m01/moment.m00);
	palm_center.x += handroi.x;
	palm_center.y += handroi.y;

	Rect roi;
	roi.x = palm_center.x - 100;
	roi.y = palm_center.y - 100;
	roi.width = 100;
	roi.height = 100;
//cout << "palm center: " << palm_center.x << " " << palm_center.y << " roi: " << roi.x << " " << roi.y << endl;
	if(roi.x<0) roi.x = 0;
	if(roi.y<0) roi.y = 0;
	if(roi.x+roi.width >=img.cols) roi.x = img.cols - roi.width;
	if(roi.y+roi.height>=img.rows) roi.y = img.rows - roi.height;*/

	Rect roi;
	roi.x = handroi.x + handroi.width/2 - box_size.width;
	roi.y = handroi.y - box_size.height/2;
	roi.width = box_size.width;
	roi.height = box_size.height;
//cout << "palm center: " << palm_center.x << " " << palm_center.y << " roi: " << roi.x << " " << roi.y << endl;
	if(roi.x<0) roi.x = 0;
	if(roi.y<0) roi.y = 0;
	if(roi.x+roi.width >=img.cols) roi.x = img.cols - roi.width;
	if(roi.y+roi.height>=img.rows) roi.y = img.rows - roi.height;

	return roi;
}

Mat getHOGClusters(Dataset &db, int version, int clusterNum)
{
	stringstream ss;
	string rootDir = db.getDataDir();
	string database = db.getDatasetName();

	ss.str("");
    ss << rootDir + "/grasp/" + database + "/version" << version << "/HOG_clusters_" << clusterNum << ".xml";
	ifstream ifile(ss.str().c_str());
    if (ifile)
    {
        ifile.close();
        Mat centers;
        FileStorage ofs;
        ofs.open(ss.str(), FileStorage::READ);
        ofs[string("centers")] >> centers;
        ofs.release();
//		cout << "[getHOGClusters]read HOG clusters from log file: " << ss.str() << endl;
        return centers;
    }
    ifile.close();
	
	Mat allFeatures;
	vector<int> trainSeqs = db.getTrainSeqs(version);
	for(int i = 0; i < (int)trainSeqs.size(); i++)
	{
		string seqName = db.getSequence(trainSeqs[i]).seqName;
		vector<Grasp> grasps = db.getGrasps(seqName);
		vector<TrackInfo> tracks = db.getTrackedHand(seqName);
		cout << seqName << ": " << (int)grasps.size() << " grasp instances " << (int)tracks.size() << " tracks\n";
		for(int g = 0; g < (int)grasps.size(); g++)
		{
			for(int f = grasps[g].startFrame; f <= grasps[g].endFrame; f++)
			{
				//read image
			    ss.str("");
			    ss << rootDir + "/img/" + seqName + "/";
			    ss << setw(8) << setfill('0') << f << ".jpg";
			    Mat img = imread(ss.str());
			    if(!img.data)
			    {
			        continue;
			    }
				ss.str("");
			    ss << rootDir + "/hand/" + seqName + "/";
			    ss << setw(8) << setfill('0') << f << ".jpg";
			    Mat hand = imread(ss.str());
			    if(!hand.data)
			    {
			        continue;
			    }
				
				//extract right-hand region according to tracked info (non-trivial task)
				TrackInfo handtrack = tracks[f];
				if(0 == handtrack.trackNum) continue; //no tracked hand in this image
				vector<RotatedRect> rRects = handtrack.rRects;
				float maxX = -1;
				int idx = -1;
				for(int j = 0; j < (int)rRects.size(); j++)
				{
					if(rRects[j].center.x > maxX)
					{
						maxX = rRects[j].center.x;
						idx = j;
					}
				}
				if(1 == (int)rRects.size() && maxX < img.cols/2)
					continue; //we consider this case as only left hand 
				Rect roi = getBox_fixed(rRects[idx], hand);				
				Rect roi_hand(roi.x+roi.width/4, roi.y, roi.width/2, roi.height);
				Mat imgroi = img(roi).clone();
		        resize(imgroi, imgroi, Size(160,80));
				Mat fvHog;
				getHOGDescriptors(imgroi, fvHog);
//cout << "[getHOGClusters] trackIdx " << f << " roi: " << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << endl;
//cout << "fvHog(0,0): " << fvHog.at<float>(0,0) << endl;
				if(checkHOG(fvHog))
				{
					cout << "[getHOGClusters] Invalid HOG descriptors: " << " seq:" << seqName << " frame:" << f << endl;
					continue;
				}
	            allFeatures.push_back(fvHog);
			}
		}
	}

	vector<int> ind = randsample(allFeatures.rows, min(20000, allFeatures.rows));
	allFeatures = indMatrixRows<float>(allFeatures, ind);
	int m = allFeatures.rows;
    int n = allFeatures.cols;
    double myepsilon = 0.00001;
    Mat labels;// = Mat::zeros(m, 1, CV_32S);
    Mat centers;// = Mat::zeros(clusterNum, n, CV_32F); 
    TermCriteria t(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, myepsilon);
    cout << "[HOG][" << m << "," << n << "] kmeans...\n";
    kmeans(allFeatures, clusterNum, labels, t, 5, KMEANS_PP_CENTERS, centers);

    ss.str("");
    ss << rootDir + "/grasp/" + database + "/version" << version << "/HOG_clusters_" << clusterNum << ".xml";
    FileStorage ifs;
    ifs.open(ss.str(), FileStorage::WRITE);
    ifs << string("centers") << centers;
    ifs.release();	
	cout << "[getHOGClusters]write HOG clusters to log file: " << ss.str() << endl;
    return centers;
}

PCA getHOGBases(Dataset &db, int version, int baseNum)
{
	stringstream ss;
	string rootDir = db.getDataDir();
	string database = db.getDatasetName();

	ss.str("");
	ss << rootDir + "/grasp/" + database + "/version" << version << "/HOG_bases_" << baseNum << ".xml";
	ifstream ifile(ss.str().c_str());
	if (ifile)
	{
		ifile.close();
		PCA pca;
		FileStorage ofs;
		ofs.open(ss.str(), FileStorage::READ);
		ofs[string("mean")] >> pca.mean;
		ofs[string("eigenvectors")] >> pca.eigenvectors;
		ofs[string("eigenvalues")] >> pca.eigenvalues;
		ofs.release();
		return pca;
	}
	ifile.close();
	
	Mat allFeatures;
	vector<int> trainSeqs = db.getTrainSeqs(version);
	for(int i = 0; i < (int)trainSeqs.size(); i++)
	{
		string seqName = db.getSequence(trainSeqs[i]).seqName;
		vector<Grasp> grasps = db.getGrasps(seqName);
		vector<TrackInfo> tracks = db.getTrackedHand(seqName);
		cout << seqName << ": " << (int)grasps.size() << " grasp instances " << (int)tracks.size() << " tracks\n";
		for(int g = 0; g < (int)grasps.size(); g++)
		{
			for(int f = grasps[g].startFrame; f <= grasps[g].endFrame; f++)
			{
				//read image
				ss.str("");
				ss << rootDir + "/img/" + seqName + "/";
				ss << setw(8) << setfill('0') << f << ".jpg";
				Mat img = imread(ss.str());
				if(!img.data)
				{
					continue;
				}
				ss.str("");
				ss << rootDir + "/hand/" + seqName + "/";
				ss << setw(8) << setfill('0') << f << ".jpg";
				Mat hand = imread(ss.str());
				if(!hand.data)
				{
					continue;
				}
				
				//extract right-hand region according to tracked info (non-trivial task)
				TrackInfo handtrack = tracks[f-1];
				if(0 == handtrack.trackNum) continue; //no tracked hand in this image
				vector<RotatedRect> rRects = handtrack.rRects;
				float maxX = -1;
				int idx = -1;
				for(int j = 0; j < (int)rRects.size(); j++)
				{
					if(rRects[j].center.x > maxX)
					{
						maxX = rRects[j].center.x;
						idx = j;
					}
				}
				if(1 == (int)rRects.size() && maxX < img.cols/2)
					continue; //we consider this case as only left hand 
					
				Rect roi = getBox_fixed(rRects[idx], hand); 			
				Mat imgroi = img(roi).clone();
				resize(imgroi, imgroi, Size(160,80));
				Mat fvHog;
				getHOGDescriptors(imgroi, fvHog);
				fvHog = fvHog.reshape(0,1);
				if(checkHOG(fvHog))
				{
					cout << "[getHOGBases] Invalid HOG descriptors: " << " seq:" << seqName << " frame:" << f << endl;
					continue;
				}
				allFeatures.push_back(fvHog);
//				if(fvHog.rows != 741)
//					cout << "WARNING: seq " << seqName << " frameid " << f << " invalid feature rows: " << fvHog.rows << " roi: " << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << endl;
			}
		}
	}

	cout << "[HOG][" << allFeatures.rows << "," << allFeatures.cols << "] PCA...\n";
//	allFeatures = allFeatures.reshape(0, allFeatures.rows*allFeatures.cols/26676);
	PCA pca(allFeatures, Mat(), CV_PCA_DATA_AS_ROW, baseNum); // principle component analysis

	ss.str("");
    ss << rootDir + "/grasp/" + database + "/version" << version << "/HOG_bases_" << baseNum << ".xml";
    FileStorage ifs;
    ifs.open(ss.str(), FileStorage::WRITE);
    ifs << string("mean") << pca.mean;
	ifs << string("eigenvectors") << pca.eigenvectors;
	ifs << string("eigenvalues") << pca.eigenvalues;
    ifs.release();	
	cout << "[getHOGBases]write HOG bases to log file: " << ss.str() << endl;
    return pca;
}

Mat getSIFTClusters(Dataset &db, int version, int clusterNum)
{
	stringstream ss;
	string rootDir = db.getDataDir();
	string database = db.getDatasetName();

	ss.str("");
    ss << rootDir + "/grasp/" + database + "/version" << version << "/SIFT_clusters_" << clusterNum << ".xml";
	ifstream ifile(ss.str().c_str());
    if (ifile)
    {
        ifile.close();
        Mat centers;
        FileStorage ofs;
        ofs.open(ss.str(), FileStorage::READ);
        ofs[string("centers")] >> centers;
        ofs.release();
//		cout << "[getSIFTClusters]read HOG clusters from log file: " << ss.str() << endl;
        return centers;
    }
    ifile.close();

    Mat training_descriptors;
	vector<int> trainSeqs = db.getTrainSeqs(version);
	for(int i = 0; i < (int)trainSeqs.size(); i++)
	{
		string seqName = db.getSequence(trainSeqs[i]).seqName;
		vector<Grasp> grasps = db.getGrasps(seqName);
		vector<TrackInfo> tracks = db.getTrackedHand(seqName);
		cout << seqName << ": " << (int)grasps.size() << " grasp instances " << (int)tracks.size() << " tracks\n";
		for(int g = 0; g < (int)grasps.size(); g++)
		{
			for(int f = grasps[g].startFrame; f <= grasps[g].endFrame; f++)
			{
				//read image
			    ss.str("");
			    ss << rootDir + "/img/" + seqName + "/";
			    ss << setw(8) << setfill('0') << f << ".jpg";
			    Mat img = imread(ss.str());
			    if(!img.data)
			    {
			        continue;
			    }
				ss.str("");
			    ss << rootDir + "/hand/" + seqName + "/";
			    ss << setw(8) << setfill('0') << f << ".jpg";
			    Mat hand = imread(ss.str());
			    if(!hand.data)
			    {
			        continue;
			    }
				
				//extract right-hand region according to tracked info (non-trivial task)
				TrackInfo handtrack = tracks[f-1];
				if(0 == handtrack.trackNum) continue; //no tracked hand in this image
				vector<RotatedRect> rRects = handtrack.rRects;
				float maxX = -1;
				int idx = -1;
				for(int j = 0; j < (int)rRects.size(); j++)
				{
					if(rRects[j].center.x > maxX)
					{
						maxX = rRects[j].center.x;
						idx = j;
					}
				}
				if(1 == (int)rRects.size() && maxX < img.cols/2)
					continue; //we consider this case as only left hand 
				Rect roi = getBox_fixed(rRects[idx], hand);

				Mat imgroi = img(roi).clone();
		        Mat fvSift = getSiftDescriptors(imgroi);
				if(checkSIFT(fvSift))
				{
					cout << "[getSIFTClusters] Invalid SIFT descriptors: " << " seq:" << seqName << " frame:" << f << endl;
					continue;
				}
		        training_descriptors.push_back(fvSift);				
			}
		}
	}

	vector<int> ind = randsample(training_descriptors.rows, min(40000, training_descriptors.rows));
	training_descriptors = indMatrixRows<float>(training_descriptors, ind);
	int m = training_descriptors.rows;
    int n = training_descriptors.cols;
    double myepsilon = 0.00001;
    Mat labels;// = Mat::zeros(m, 1, CV_32S);
    Mat centers;// = Mat::zeros(clusterNum, n, CV_32F); 
    TermCriteria t(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, myepsilon);
    cout << "[SIFT][" << m << "," << n << "] kmeans...\n";
    kmeans(training_descriptors, clusterNum, labels, t, 5, KMEANS_PP_CENTERS, centers);

    ss.str("");
    ss << rootDir + "/grasp/" + database + "/version" << version << "/SIFT_clusters_" << clusterNum << ".xml";
    FileStorage ifs;
    ifs.open(ss.str(), FileStorage::WRITE);
    ifs << string("centers") << centers;
    ifs.release();	
	cout << "[getSIFTClusters]write SIFT clusters to log file: " << ss.str() << endl;
    return centers;
}

cvGrasp :: cvGrasp(string &cfgName)
{
	ConfigFile cfg(cfgName);
	_cfg = cfg;
	cout << "finish parsing configuration file\n";
}

int cvGrasp :: getWorkDir()
{
	stringstream ss;

	if(_cfg.keyExists("videoname"))
	{
		string vidname = _cfg.getValueOfKey<string>("videoname");
		while(vidname.find(" ") != vidname.npos)
		{
			ss << vidname.substr(0, vidname.find(" "));
			ss << "_";
			vidname.erase(0, vidname.find(" "));
			vidname.erase(0, vidname.find_first_not_of(" "));
		}
		ss << vidname;
	}
	else
	{
		return ERR_CFG_VIDEONAME_NONEXIST;
	}

	if(_cfg.keyExists("rootname"))
	{
		string rootname = _cfg.getValueOfKey<string>("rootname");
		_workDir = rootname + "/grasp/" + ss.str();
	}
	else
	{
		return ERR_CFG_ROOTNAME_NONEXIST;
	}

	return 0;
}

int cvGrasp :: getFeatureType()
{
	if(_cfg.keyExists("feature"))
	{
		string feat = _cfg.getValueOfKey<string>("feature");
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
	return 0;
}

int cvGrasp :: prepareData()
{
	int stat = 0;

	stat = _dp.initialize(_cfg);
	if(stat) return stat;

	stat = _dp.getGraspFromGTEA();
	if(stat) return stat;

	return 0;
}

int cvGrasp :: computeFeature()
{
	int stat = 0;

	stat = _fe.initialize(_cfg);
	if(stat) return stat;

	//get work directory
	stat = getWorkDir();
	if(stat) return stat;

	stat = _fe.compute(_workDir);
	if(stat) return stat;

	return 0;
}

int cvGrasp :: hierarchyCluster()
{
	int stat = 0;

	stat = _hc.initialize(_cfg);
	if(stat) return stat;

	//get work directory
	stat = getWorkDir();
	if(stat) return stat;

	//get feature types used for clustering
	stat = getFeatureType();
	if(stat) return stat;

	stat = _hc.cluster(_workDir, _featureType);
	if(stat) return stat;

	return 0;
}

/*********************** Supervised Part ****************************/
int cvGrasp :: trainDataGraspClassifier_interval(Dataset &db, int version, string seqName, Grasp &grasp, vector<TrackInfo> &tracks, Mat &data, vector<GraspNode> &nodes)
{
	int stat = 0;
	int startFrame = grasp.startFrame;
	int endFrame = grasp.endFrame;
	stringstream ss;
	string rootDir = db.getDataDir();
	string dbName = db.getDatasetName();
	string seq_ = seqName;
	seq_.erase(seq_.begin(), seq_.begin() + seq_.find_first_of("/") + 1);

	//check if there already exists data file
	char dataFile[300];
	dataFile[0] = 0;
	sprintf(dataFile, "%s/grasp/%s/version%d/temp/DataGraspClassifier", rootDir.c_str(), dbName.c_str(), version);
	sprintf(dataFile, "%s_%s_%d_%d.xml", dataFile, seq_.c_str(), startFrame, endFrame);
	ifstream ifile(dataFile);
    if(ifile)
    {
        ifile.close();
        FileStorage ofs;
        ofs.open(string(dataFile), FileStorage::READ);
        ofs[string("Data")] >> data;
		FileNode feature = ofs[string("Node")];
		for(FileNodeIterator it = feature.begin(); it != feature.end(); it++)
		{
			GraspNode node;
			(*it)["graspType"] >> node.graspType;
			(*it)["seqName"] >> node.seqName;
			node.frameid = (int)(*it)["frameid"];
			vector<int> roi;
			(*it)["roi"] >> roi;
			node.roi = Rect(roi[0],roi[1],roi[2],roi[3]);
			nodes.push_back(node);
		}
        ofs.release();
        return 0;
    }
    ifile.close();
		
	Mat cluster_hog = getHOGClusters(db, version, 100);
	Mat cluster_sift = getSIFTClusters(db, version, 100);
	PCA pca = getHOGBases(db, version, 100);
	for(int i = startFrame; i <= endFrame; i++)
	{
	    //read hand probability image
	    ss.str("");
	    ss << rootDir + "/hand/" + seqName + "/";
	    ss << setw(8) << setfill('0') << i << ".jpg";
	    Mat hand = imread(ss.str());
	    if(!hand.data)
	    {
	        continue;
	    }
		
		//extract right-hand region according to tracked info
		TrackInfo handtrack = tracks[i-1];
		if(0 == handtrack.trackNum) continue; //no tracked hand in this image
		vector<RotatedRect> rRects = handtrack.rRects;
		float maxX = -1;
		int idx = -1;
		for(int j = 0; j < (int)rRects.size(); j++)
		{
			if(rRects[j].center.x > maxX)
			{
				maxX = rRects[j].center.x;
				idx = j;
			}
		}
		if(1 == (int)rRects.size() && maxX < hand.cols/2)
			continue; //we consider this case as only left hand 
		Rect roi = getBox_fixed(rRects[idx], hand);
		Rect roi_hand(roi.x+roi.width/4, roi.y, roi.width/2, roi.height);

		//feature extraction
		Mat fv;
		
		Mat handroi = hand(roi).clone();
		//extract shape features
/*		Mat f_shape;
		stat = _fe.getFeature_Shape(handroi, f_shape);
		if(stat)
		{
			cout << "[trainDataGraspClassifier_interval] bad Shape feature: " << seqName << " frameno " << i << endl;
			continue;
		}
		f_shape = f_shape.t();
		fv.push_back(f_shape);*/

		//read hand image
	    ss.str("");
	    ss << rootDir + "/img/" + seqName + "/";
	    ss << setw(8) << setfill('0') << i << ".jpg";
	    Mat img = imread(ss.str());
	    if(!img.data)
	    {
	        continue;
	    }
		Mat imgroi = img(roi).clone();

		//extract sift features
/*		Mat f_sift;
//		Rect roi_obj = getBox_object(roi, hand);
//		imgroi = img(roi_obj).clone();
		stat = _fe.getFeature_SIFT_BOW(imgroi, cluster_sift, f_sift);
		if(stat)
		{
			cout << "[trainDataGraspClassifier_interval] bad SIFT feature: " << seqName << " frameno " << i << endl;
			continue;
		}
		f_sift = f_sift.t();
		fv.push_back(f_sift);*/

		//extract hog features
		Mat f_hog;		
		imgroi = img(roi).clone();
        resize(imgroi, imgroi, Size(160,80));		
//		stat = _fe.getFeature_HOG_BOW(imgroi, cluster_hog, f_hog);
//		stat = _fe.getFeature_HandHOG(imgroi, handroi, f_hog);
		stat = _fe.getFeature_HOG(imgroi, f_hog);
//		stat = _fe.getFeature_HOG_PCA(imgroi, f_hog, pca);
		if(stat)
		{
			cout << "[trainDataGraspClassifier_interval] bad HOG feature: " << seqName << " frameno " << i << endl;
			exit(-1);
			continue;
		}
		f_hog = f_hog.t();
		fv.push_back(f_hog);
		
		fv = fv.t();
		data.push_back(fv);

		GraspNode gNode;
		gNode.seqName = seq_;
		gNode.frameid = i;
		gNode.roi = roi;
		gNode.graspType = grasp.graspType;
		nodes.push_back(gNode);

	}

	FileStorage ifs;
    ifs.open(string(dataFile), FileStorage::WRITE);
    ifs << "Data" << data;
	ifs << "Node" << "[";
	for(int i = 0; i < (int)nodes.size(); i++)
	{
		ifs << "{";
		ifs << "graspType" << nodes[i].graspType;
		ifs << "seqName" << nodes[i].seqName;
		ifs << "frameid" << nodes[i].frameid;
		ifs << "roi" << "[";
		ifs << nodes[i].roi.x << nodes[i].roi.y << nodes[i].roi.width << nodes[i].roi.height;
		ifs << "]" << "}";
	}
	ifs << "]";
    ifs.release();
	return 0;
}

int cvGrasp :: trainDataGraspClassifier(string datasetName, int version, Grasp grasp, vector<Grasp> freqGrasps, Mat &trainData, Mat &labels, vector<GraspNode> &trainNodes)
{
	Dataset db = dataset_setup(datasetName);
	vector<int> trainSeqs = db.getTrainSeqs(version);
	for(int i = 0; i < (int)trainSeqs.size(); i++)
	{
		string seqName = db.getSequence(trainSeqs[i]).seqName;
		vector<Grasp> grasps = db.getGrasps(seqName);
		vector<TrackInfo> tracks = db.getTrackedHand(seqName);
		for(int g = 0; g < (int)grasps.size(); g++)
		{
			bool isFreq = false;
			for(int gf = 0; gf < (int)freqGrasps.size(); gf++)
			{
				if(grasps[g].graspType == freqGrasps[gf].graspType)
				{
					isFreq = true;
					break;
				}
			}
			if(!isFreq)
				continue;

			bool isPos = (grasps[g].graspType == grasp.graspType);
			Mat feature;
			vector<GraspNode> nodes;
			trainDataGraspClassifier_interval(db, version, seqName, grasps[g], tracks, feature, nodes);
			if(0 == feature.rows)
			{
//				cout << "[trainDataGraspClassifier] no feature extracted: " << seqName << " frameno " << grasps[g].startFrame << "-" << grasps[g].endFrame << endl;
				continue;
			}
			trainData.push_back(feature);
			for(int m = 0; m < (int)nodes.size(); m++)
				trainNodes.push_back(nodes[m]);
			Mat label = Mat::ones(feature.rows,1,CV_32F);
			if(!isPos)
				label = label*0;
			labels.push_back(label);
		}
	}
	return 0;
}

template<class T>
vector<T> cvGrasp :: trainGraspClassifiers(string datasetName, int version, vector<Grasp> freqGrasps)
{
	Dataset db = dataset_setup(datasetName);
	string dbName = db.getDatasetName();
	string dataDir = db.getDataDir();
	vector<T> trainers((int)freqGrasps.size());
	for(int a = 0; a < (int)freqGrasps.size(); a++)
	{
		T trainer;
		cout << a << ": " << freqGrasps[a].graspType << endl;

		//check if there already exists model file
		char modelFile[300];
		modelFile[0] = 0;
		sprintf(modelFile, "%s/grasp/%s/version%d/model/graspClassifier_svm_%s.xml", dataDir.c_str(), dbName.c_str(), version, freqGrasps[a].graspType.c_str());
		ifstream ifile(modelFile);
	    if (ifile)
	    {
	        ifile.close();
			trainers[a].load(modelFile);
	        continue;
	    }
	    ifile.close();

		char tempdataFile[300];
		tempdataFile[0] = 0;
		sprintf(tempdataFile, "%s/grasp/%s/version%d/model/graspClassifier_svm_tempdata_%s", dataDir.c_str(), dbName.c_str(), version, freqGrasps[a].graspType.c_str());

		char tempmodelFile[300];
		tempmodelFile[0] = 0;
		sprintf(tempmodelFile, "%s/grasp/%s/version%d/model/graspClassifier_svm_tempmodel_%s", dataDir.c_str(), dbName.c_str(), version, freqGrasps[a].graspType.c_str());
		
		Mat trainData, labels;
		vector<GraspNode> trainNodes;
		trainDataGraspClassifier(datasetName, version, freqGrasps[a], freqGrasps, trainData, labels, trainNodes);
		downSampleTrainData(trainData, labels, trainNodes, 5, 1);
		filterTrainData(db, version, trainData, labels, trainNodes, true); //filter out bad hand detection
		Size pn = arrange_pos_neg(trainData, labels); //arrange positive/negative samples		
		
		if(pn.width > 0 && pn.height > 0)
		{			
			balance_pos_neg(trainData, labels, pn.width, pn.height, (int)freqGrasps.size());
			trainer.train(trainData, labels, string(tempdataFile), string(tempmodelFile));
		}
		else
			cout << "no trainer obtained for grasp: " << freqGrasps[a].graspType << endl;
		trainer.save(modelFile);
		trainers[a].load(modelFile);
/*
		char command[300];
		command[0] = 0;
		sprintf(command, "rm -f %s", tempdataFile);
		system(command);

		command[0] = 0;
		sprintf(command, "rm -f %s", tempmodelFile);
		system(command);*/
	}
	
	return trainers;
}

int cvGrasp :: testDataGraspClassifier_interval(Dataset &db, int version, string seqName, Grasp &grasp, vector<TrackInfo> &tracks, Mat &data, vector<GraspNode> &nodes)
{
	//same as training data extraction
	trainDataGraspClassifier_interval(db, version, seqName, grasp, tracks, data, nodes);
	return 0;
}

template<class T>
pair<Mat, Mat> cvGrasp :: testGraspClassifier(string datasetName, int version, vector<T> trainers, vector<Grasp> freqGrasps)
{
	Mat classifications, testData, labels;
	vector<GraspNode> testNodes;

	//classification: one result for one grasp interval
	Dataset db = dataset_setup(datasetName);	
	vector<int> testSeqs = db.getTestSeqs(version);
	for(int i = 0; i < (int)testSeqs.size(); i++)
	{
		string seqName = db.getSequence(testSeqs[i]).seqName;
		vector<Grasp> grasps = db.getGrasps(seqName);
		vector<TrackInfo> tracks = db.getTrackedHand(seqName);
		cout << seqName << ": " << (int)grasps.size() << " grasp instances " << (int)tracks.size() << " tracks\n";
		int fgNum = -1;
		for(int g = 0; g < (int)grasps.size(); g++)
		{
			bool isFreq = false;
			for(int gf = 0; gf < (int)freqGrasps.size(); gf++)
			{
				if(grasps[g].graspType == freqGrasps[gf].graspType)
				{
					isFreq = true;
					fgNum = gf;
					break;
				}
			}
			if(!isFreq)
				continue;
			
			Mat feature;
			vector<GraspNode> nodes;
			testDataGraspClassifier_interval(db, version, seqName, grasps[g], tracks, feature, nodes);
//			cout << "[" << grasps[g].graspType << "](" << grasps[g].startFrame << "-" << grasps[g].endFrame << "):";
			if(0 == feature.rows)
			{
//				cout << " no hand detected!\n";
				continue;
			}
//			cout << " grasp number " << fgNum << " feature dim: " << testData.rows << "x" << testData.cols << endl;
			testData.push_back(feature);
			for(int m = 0; m < (int)nodes.size(); m++)
				testNodes.push_back(nodes[m]);			

			Mat label = Mat::ones(feature.rows,1,CV_32S);
			label = label*fgNum;
			labels.push_back(label);

			
		}
		
	}

	downSampleTestData(testData, labels, testNodes, 5, 1, true);//divide into train and remaining; for checking overfitting
	filterTestData(db, version, testData, labels, testNodes, true);//filter out bad hand detection
	
    for(int m = 0; m < (int)trainers.size(); m++)
    {
        Mat output = trainers[m].predict_prob(testData);
        output = output.t();
		classifications.push_back(output);
    }
    classifications = classifications.t();

	//record false positive
	recordFP(db, version, labels, classifications, testNodes, freqGrasps);
	writeFP2Html(db, version, freqGrasps);

	//save classification score to csv file
	string dataDir = db.getDataDir();
	string dbName = db.getDatasetName();
	char filename[500];
	filename[0] = 0;
	sprintf(filename, "%s/grasp/%s/version%d/score.csv", dataDir.c_str(), dbName.c_str(), version);
	ofstream outfile(filename);
	if(!outfile.is_open())
	{
		cout << "[testGraspClassifier] failed to open file: " << filename << endl;
		exit(-1);
	}		
	for(int r = 0; r < classifications.rows; r++)
	{
		outfile << testNodes[r].seqName << "," << testNodes[r].frameid << ",";
		for(int c = 0; c < classifications.cols; c++)
			outfile << classifications.at<float>(r,c) << ",";
		outfile << endl;
	}
	outfile.close();
			
	pair<Mat, Mat> answers;
	answers.first = classifications;
	answers.second = labels;
	return answers;
}

int cvGrasp :: procGTEAGrasp()
{
	int stat = 0;

	stat = _dp.initialize(_cfg);
	if(stat) return stat;

	stat = _dp.getGraspFromGTEA();
	if(stat) return stat; 

	stat = _fe.initialize(_cfg);
	if(stat) return stat;

	stat = _fe.getFeatureAllFromGTEA(_dp._handInfo);
	if(stat) return stat;

	stat = _hc.initialize(_cfg);
	if(stat) return stat;

	stat = _hc.getHiClusterAllFromGTEA(_fe._featureInfo);
	if(stat) return stat;
	
	return 0;
}

int cvGrasp :: procIntelGrasp()
{
	int stat = 0;

	stat = _dp.initialize(_cfg);
	if(stat) return stat;

	stat = _dp.getGraspFromIntel();
	if(stat) return stat; 

	stat = _fe.initialize(_cfg);
	if(stat) return stat;

	stat = _fe.getFeatureAllFromIntel(_dp._handInfo);
	if(stat) return stat;

	stat = _hc.initialize(_cfg);
	if(stat) return stat;

	stat = _hc.getClusterAllFromIntel(_fe._featureInfo);
	if(stat) return stat;
	
	return 0;
}

int cvGrasp :: procYaleGraspClassify()
{
	int stat = 0;
	stringstream ss;
	int version = _cfg.getValueOfKey<int>("version");
	string datasetName("Yale");
	Dataset db = dataset_setup(datasetName);

	stat = _dp.initialize(_cfg);
	if(stat) return stat;

	stat = _fe.initialize(_cfg);
	if(stat) return stat;

	string dataDir = db.getDataDir();
	string dbName = db.getDatasetName();
	ss.str("");
	ss << "mkdir " + dataDir + "/grasp/" + dbName;
	system(ss.str().c_str());
	ss.str("");
	ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version;
	system(ss.str().c_str());
	ss.str("");
	ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp";
	system(ss.str().c_str());
	ss.str("");
	ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/model";
	system(ss.str().c_str());
	
//	vector<Grasp> freqGrasps = db.getFrequentGrasps(version, 3, 1);
	vector<Grasp> freqGrasps = db.getfrequentGraspsInTraining(version, 1);
	for(int a = 0; a < (int)freqGrasps.size(); a++)
	{
	    cout << a << ": " << freqGrasps[a].graspType;
	    cout << " ";
	    int freqTrain = db.getGraspFreqInTraining(freqGrasps[a], version);
	    int freqTest = db.getGraspFreqInTesting(freqGrasps[a], version);
	    cout << "(" << freqTrain << ", " << freqTest << ")";
	    cout << endl;

		ss.str("");
		ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType;
		system(ss.str().c_str());

		//for restoring false positive
		ss.str("");
		ss << "rm -r " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/FP";
		system(ss.str().c_str());
		ss.str("");
		ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/FP";
		system(ss.str().c_str());

		//for restoring true positive
		ss.str("");
		ss << "rm -r " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/TP";
		system(ss.str().c_str());
		ss.str("");
		ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/TP";
		system(ss.str().c_str());

		//for restoring training samples
		ss.str("");
		ss << "rm -r " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/train";
		system(ss.str().c_str());
		ss.str("");
		ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/train";
		system(ss.str().c_str());
		//for restoring test samples
		ss.str("");
		ss << "rm -r " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/test";
		system(ss.str().c_str());
		ss.str("");
		ss << "mkdir " + dataDir + "/grasp/" + dbName + "/version" << version << "/temp/" + freqGrasps[a].graspType + "/test";
		system(ss.str().c_str());
  	}
	
	vector<LcSVM> trainers = trainGraspClassifiers<LcSVM>(datasetName, version, freqGrasps);	

	cout << "Testing now..." << endl;

	pair<Mat,Mat> results = testGraspClassifier<LcSVM>(datasetName, version, trainers, freqGrasps);
	cout << "Done with the results" << endl;

	
	cout << "Showing Confusion Matrix" << endl;
	Mat confMatrix = classificationToConfusionMatrix(results.first, results.second);		

	float accuracy = getAccuracy(confMatrix, freqGrasps);
	cout << "Grasp recognition accuracy for " << confMatrix.rows << " classes is: " << accuracy << endl;
	Mat confMatrix_r(confMatrix.size(), CV_32F);
	float F1 = getF1(confMatrix, confMatrix_r, freqGrasps);
	cout << "Average F1 score for " << confMatrix.rows << " classifiers is: " << F1 << endl;

	//save confusion matrix to csv file
	ss.str("");
	ss << dataDir + "/grasp/" + dbName + "/version" << version << "/confusion.csv";
	ofstream outfile(ss.str().c_str());
	if(!outfile.is_open())
	{
		cout << "[procYaleGraspClassify] failed to open file: " << ss.str() << endl;
		return -1;
	}		
	for(int r = 0; r < confMatrix_r.rows; r++)
	{
		for(int c = 0; c < confMatrix_r.cols; c++)
			outfile << confMatrix_r.at<float>(r,c) << ",";
		outfile << endl;
	}
	outfile.close();

	Mat correlation = getCorrelation(confMatrix);
	//save correlation matrix to csv file
	ss.str("");
	ss << dataDir + "/grasp/" + dbName + "/version" << version << "/correlation.csv";
	outfile.open(ss.str().c_str());
	if(!outfile.is_open())
	{
		cout << "[procYaleGraspClassify] failed to open file: " << ss.str() << endl;
		return -1;
	}		
	for(int r = 0; r < correlation.rows; r++)
	{
		for(int c = 0; c < correlation.cols; c++)
			outfile << correlation.at<float>(r,c) << ",";
		outfile << endl;
	}
	outfile.close();
	
	return 0;
}

int main(int argc, char** argv)
{
	int stat = 0;
	string operation;
	string cfgName;

	if(argc < 2)
	{
		operation = "all";
		cfgName = "all.config";
	}
	else
	{
		int k = 1;
		while(k < argc)
		{
			if("-operate" == (string)argv[k]) operation = (string)argv[++k];
			else if("-cfgname" == (string)argv[k]) cfgName = (string)argv[++k];
			else stat = ERR_CLI_PARAM_INVALID;
			k++;
		}
	}

	if(stat)
	{
		cout << "./cvGrasp -operate <string> -cfgname <string>\n";
		return -1;
	}

	//obtain instance of class cvGrasp, and parse configuration file
	cvGrasp grasp(cfgName);

	if("data" == operation)
	{
		stat = grasp.prepareData();
	}
	else if("feature" == operation)
	{
		stat = grasp.computeFeature();
	}
	else if("cluster" == operation)
	{
		stat = grasp.hierarchyCluster();
	}
	else if("Intel" == operation)
	{
		stat = grasp.procIntelGrasp();
	}
	else if("GTEA" == operation)
	{
		stat = grasp.procGTEAGrasp();
	}
	else if("Yale" == operation)
	{
		stat = grasp.procYaleGraspClassify();
	}
	else
	{
		stat = ERR_CLI_VALUE_INVALID;
	}
	
	if(0 == stat)
	{
		cout << "finished!\n";
		return 0;
	}
	else if(ERR_CLI_VALUE_INVALID == stat)
	{
		cout << "candidate mode for -operate:" << endl;
		cout << "  <data> prepare input data" << endl;
		cout << "  <feature> extract feature" << endl;
		cout << "  <cluster> group grasptype" << endl;
		cout << "  <all> do all the work" << endl;
		cout << "  <Intel> group grasptype for Intel_Objects_Egocentric dataset" << endl;
		cout << "  <GTEA> group grasptype for GTEA_plus dataset" << endl;
		cout << "  <Yale> classify/group grasptype for Yale Human Grasp dataset" << endl;
		return -1;
	}
	else
	{
		cout << "failed with error: <" << stat << ">\n";
		return -1;
	}

}
