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

void Grasp::removeBadChars()
{
	for(int i = (int)graspType.length()-1; i >=0; i--)
	{
		if(graspType[i] == ' ')
			graspType[i] = '_';
		if(graspType[i] == '-')
			graspType.erase(graspType.begin()+i);
	}
}
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
vector<Action> readActionAnnotations(const char* filename)
{
    vector<Action> actions;

    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[readActionAnnotations] failed to open file: " << filename << endl;
        exit(1);
    }
    int limit = 500;
    char line[limit];
    while(infile.good())
    {
        line[0] = 0;
        infile.getline(line, limit);
        if(!line[0])
            continue;
        cout << "read line: " << line << endl;
        Action tempAction;
        char * tier = strtok(line, "\t");
		tier = tier;
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
vector<int> readObjectAnnotations(const char* filename)
{
    vector<int> objects;
    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[readObjectAnnotations] failed to open file: " << filename << endl;
        exit(1);
    }
    int limit = 500;
    char line[limit];
    while(infile.good())
    {
        line[0] = 0;
        infile.getline(line, limit);
        if(!line[0])
            continue;
        int objectId = atoi(line);
        objects.push_back(objectId);
    }

    infile.close();
    return objects;
}

/*****************************************************************************************/
vector<Grasp> readGraspAnnotations(const char* filename)
{
    vector<Grasp> grasps;

    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[readGraspAnnotations] failed to open file: " << filename << endl;
        exit(1);
    }
    int limit = 500;
    char line[limit];
    while(infile.good())
    {
        line[0] = 0;
        infile.getline(line, limit);
        if(!line[0])
            continue;
		if('#' == line[0]) // comment
			continue; 
        Grasp tempGrasp;
        char* graspType = strtok(line, "\t");
        char* startFrame = strtok(NULL, "\t");
        char* endFrame = strtok(NULL, "\t");
        tempGrasp.graspType = string(graspType);
        tempGrasp.startFrame = atoi(startFrame);
        tempGrasp.endFrame = atoi(endFrame);
		tempGrasp.removeBadChars();
        grasps.push_back(tempGrasp);
    }

    return grasps;
}

vector<TrackInfo> readTrackLog(const char* filename)
{
    vector<TrackInfo> tracks(0);
    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[readTrackLog] failed to open file: " << filename << endl;
        exit(1);
    }
    int limit = 500;
    char line[limit];
    while(infile)
    {
        line[0] = 0;
        infile.getline(line, limit);
        if(!infile)
            break;
        TrackInfo temp;
        char* trackNum = strtok(line, "\t");
        temp.trackNum = atoi(trackNum);
        for(int i = 0; i < temp.trackNum; i++)
        {
            RotatedRect rRect;
            char* centerX = strtok(NULL, "\t(,)");
            rRect.center.x = atof(centerX);
            char* centerY = strtok(NULL, "\t(,)");
            rRect.center.y = atof(centerY);
            char* axisL = strtok(NULL, "\t(,)");
            rRect.size.height = atof(axisL);
            char* axisS = strtok(NULL, "\t(,)");
            rRect.size.width = atof(axisS);
            char* angle = strtok(NULL, "\t(,)");
            rRect.angle = atof(angle);
            temp.rRects.push_back(rRect);
        }
        tracks.push_back(temp);
    }
    infile.close();

    return tracks;
}

vector<HandInfo> readHandInfoLog(const char* filename, int objectId)
{
    vector<HandInfo> handInfo;
    ifstream infile(filename, ios::in);
    if(!infile)
    {
        cout << "[readHandInfoLog] failed to open file: " << filename << endl;
        exit(1);
    }
    int limit = 500;
    char line[limit];
    while(infile)
    {
        line[0] = 0;
        infile.getline(line, limit);
        if(!infile)
            break;
        HandInfo temp;
        temp.objectId = objectId;
        char* seqNum = strtok(line, "\t");
        temp.seqNum= atoi(seqNum);
        char* frameNum = strtok(NULL, "\t");
        temp.frameNum= atoi(frameNum);
        char* handState = strtok(NULL, "\t");
        temp.handState= atoi(handState);
        for(int i = 0; i < 2; i++)
        {
            Rect box;
            char* centerX = strtok(NULL, "\t(,)");
            box.x = atof(centerX);
            char* centerY = strtok(NULL, "\t(,)");
            box.y = atof(centerY);
            char* width = strtok(NULL, "\t(,)");
            box.width = atof(width);
            char* height = strtok(NULL, "\t(,)");
            box.height = atof(height);
            temp.box[i] = box;
        }
        handInfo.push_back(temp);
    }
    infile.close();

    return handInfo;

}

Dataset dataset_setup_Yale(string datasetName)
{
    Dataset db(datasetName, "/home/cai-mj/_GTA", 25, "jpg");
	//mechanist 2
    db.addSequence(1, datasetName+"/001", 1, 15001);
	db.addSequence(2, datasetName+"/002", 1, 15001);
	db.addSequence(3, datasetName+"/003", 1, 8501);
	db.addSequence(4, datasetName+"/004", 1, 15001);
	db.addSequence(5, datasetName+"/005", 1, 4901);
	db.addSequence(6, datasetName+"/006", 1, 8151);
	db.addSequence(7, datasetName+"/007", 1, 12751);
	db.addSequence(8, datasetName+"/008", 1, 15001);
	db.addSequence(10, datasetName+"/010", 1, 4138);
	db.addSequence(11, datasetName+"/011", 1, 14676);
	db.addSequence(12, datasetName+"/012", 1, 15001);
	db.addSequence(13, datasetName+"/013", 1, 15001);
	db.addSequence(14, datasetName+"/014", 1, 15001);
	db.addSequence(15, datasetName+"/015", 1, 15001);
	db.addSequence(16, datasetName+"/016", 1, 15001);
	db.addSequence(17, datasetName+"/017", 1, 15001);
	db.addSequence(18, datasetName+"/018", 1, 15001);
	db.addSequence(19, datasetName+"/019", 1, 4426);
	db.addSequence(20, datasetName+"/020", 1, 12651);

	//housekeeper 1
	db.addSequence(41, datasetName+"/041", 1, 44976);
	//mechanist 1
    db.addSequence(87, datasetName+"/087", 1, 44976);

	//new controlled data
	db.addSequence(201, datasetName+"/201", 1, 3387);
	db.addSequence(202, datasetName+"/202", 1, 2878);
	db.addSequence(203, datasetName+"/203", 1, 7182);
	db.addSequence(204, datasetName+"/204", 1, 2829);
	db.addSequence(205, datasetName+"/205", 1, 1751);
	db.addSequence(206, datasetName+"/206", 1, 4382);
	db.addSequence(207, datasetName+"/207", 1, 1487);
	db.addSequence(208, datasetName+"/208", 1, 1517);
	db.addSequence(211, datasetName+"/211", 1, 3782);
	db.addSequence(212, datasetName+"/212", 1, 3608);
	db.addSequence(213, datasetName+"/213", 1, 4165);
	db.addSequence(214, datasetName+"/214", 1, 3225);
	db.addSequence(215, datasetName+"/215", 1, 3363);
	db.addSequence(216, datasetName+"/216", 1, 3746);
	db.addSequence(221, datasetName+"/221", 1, 2068);
	db.addSequence(222, datasetName+"/222", 1, 2859);
	db.addSequence(223, datasetName+"/223", 1, 3747);
	db.addSequence(224, datasetName+"/224", 1, 2554);
	db.addSequence(225, datasetName+"/225", 1, 2662);
	db.addSequence(226, datasetName+"/226", 1, 3794);
	db.addSequence(231, datasetName+"/231", 1, 3351);
	db.addSequence(232, datasetName+"/232", 1, 3153);
	db.addSequence(233, datasetName+"/233", 1, 4256);
	db.addSequence(234, datasetName+"/234", 1, 2709);
	db.addSequence(235, datasetName+"/235", 1, 3100);
	db.addSequence(236, datasetName+"/236", 1, 3993);

    vector<int> trainS;
    trainS.push_back(1);
	trainS.push_back(3);
	trainS.push_back(5);
	trainS.push_back(7);
	trainS.push_back(11);
	trainS.push_back(13);
	trainS.push_back(15);
	trainS.push_back(17);
	trainS.push_back(19);
	trainS.push_back(2);
	trainS.push_back(4);
	trainS.push_back(6);
	trainS.push_back(8);
	trainS.push_back(10);
	trainS.push_back(12);
	trainS.push_back(14);
	trainS.push_back(16);
	trainS.push_back(18);
	trainS.push_back(20);
    db.setTrainSeqs(1, trainS);

    vector<int> testS;
	testS.push_back(87);
    db.setTestSeqs(1, testS);

	trainS.clear();
    trainS.push_back(1);
	trainS.push_back(3);
	trainS.push_back(5);
	trainS.push_back(7);
	trainS.push_back(11);
	trainS.push_back(13);
	trainS.push_back(15);
	trainS.push_back(17);
	trainS.push_back(19);
	trainS.push_back(2);
	trainS.push_back(4);
	trainS.push_back(6);
	trainS.push_back(8);
	trainS.push_back(10);
	trainS.push_back(12);
	trainS.push_back(14);
	trainS.push_back(16);
	trainS.push_back(18);
	trainS.push_back(20);
    db.setTrainSeqs(2, trainS);

    testS.clear();
    testS.push_back(41);
    db.setTestSeqs(2, testS);

	trainS.clear();
    trainS.push_back(1);
	trainS.push_back(3);
	trainS.push_back(5);
	trainS.push_back(7);
	trainS.push_back(11);
	trainS.push_back(13);
	trainS.push_back(15);
	trainS.push_back(17);
	trainS.push_back(19);
	trainS.push_back(2);
	trainS.push_back(4);
	trainS.push_back(6);
	trainS.push_back(8);
	trainS.push_back(10);
	trainS.push_back(12);
	trainS.push_back(14);
	trainS.push_back(16);
	trainS.push_back(18);
	trainS.push_back(20);
    db.setTrainSeqs(3, trainS);

    testS.clear();
    testS.push_back(1);
	testS.push_back(3);
	testS.push_back(5);
	testS.push_back(7);
	testS.push_back(11);
    testS.push_back(13);
	testS.push_back(15);
	testS.push_back(17);
	testS.push_back(19);
	testS.push_back(2);
	testS.push_back(4);
	testS.push_back(6);
	testS.push_back(8);
	testS.push_back(10);
    testS.push_back(12);
	testS.push_back(14);
	testS.push_back(16);
	testS.push_back(18);
	testS.push_back(20);
    db.setTestSeqs(3, testS);

	trainS.clear();
	trainS.push_back(201);
	trainS.push_back(202);
	trainS.push_back(203);
	trainS.push_back(208);
	trainS.push_back(211);
	trainS.push_back(212);
	trainS.push_back(213);
	trainS.push_back(221);
	trainS.push_back(222);
	trainS.push_back(223);
/*	trainS.push_back(231);
	trainS.push_back(232);
	trainS.push_back(233);*/
	db.setTrainSeqs(4, trainS);

	testS.clear();
/*	testS.push_back(201);
	testS.push_back(202);
	testS.push_back(203);
	testS.push_back(208);
	testS.push_back(211);
	testS.push_back(212);
	testS.push_back(213);
	testS.push_back(221);
	testS.push_back(222);
	testS.push_back(223);*/
	testS.push_back(231);
	testS.push_back(232);
	testS.push_back(233);
	db.setTestSeqs(4, testS);

	trainS.clear();
	trainS.push_back(201);
	trainS.push_back(202);
	trainS.push_back(203);
	trainS.push_back(208);
	db.setTrainSeqs(5, trainS);

	testS.clear();
	testS.push_back(204);
	testS.push_back(205);
	testS.push_back(206);
	testS.push_back(207);
	db.setTestSeqs(5, testS);

	trainS.clear();
	trainS.push_back(201);
	trainS.push_back(202);
	trainS.push_back(203);
//	trainS.push_back(204);
//	trainS.push_back(205);
//	trainS.push_back(206);
//	trainS.push_back(207);
	trainS.push_back(208);
/*	trainS.push_back(211);
	trainS.push_back(212);
	trainS.push_back(213);
	trainS.push_back(214);
	trainS.push_back(215);
	trainS.push_back(216);
	trainS.push_back(221);
	trainS.push_back(222);
	trainS.push_back(223);
	trainS.push_back(224);
	trainS.push_back(225);
	trainS.push_back(226);
	trainS.push_back(231);
	trainS.push_back(232);
	trainS.push_back(233);
	trainS.push_back(234);
	trainS.push_back(235);
	trainS.push_back(236);*/
	db.setTrainSeqs(6, trainS);

	testS.clear();
	testS.push_back(201);
	testS.push_back(202);
	testS.push_back(203);
//	testS.push_back(204);
//	testS.push_back(205);
//	testS.push_back(206);
//	testS.push_back(207);
	testS.push_back(208);
/*	testS.push_back(211);
	testS.push_back(212);
	testS.push_back(213);
	testS.push_back(214);
	testS.push_back(215);
	testS.push_back(216);
	testS.push_back(221);
	testS.push_back(222);
	testS.push_back(223);
	testS.push_back(224);
	testS.push_back(225);
	testS.push_back(226);
	testS.push_back(231);
	testS.push_back(232);
	testS.push_back(233);
	testS.push_back(234);
	testS.push_back(235);
	testS.push_back(236);*/
	db.setTestSeqs(6, testS);

    return db;
}

Dataset dataset_setup_GTEA(string datasetName)
{
    Dataset db(datasetName, "/home/cai-mj/_GTA", 24, "jpg");
    db.addSequence(1, datasetName+"/Alireza_American", 1, 19844);

    vector<int> trainS;
    trainS.push_back(1);
    db.setTrainSeqs(1, trainS);

    return db;
}

Dataset dataset_setup_Intel(string datasetName)
{
    Dataset db(datasetName, "/home/cai-mj/_GTA", 1, "jpg");
    db.addSequence(1, datasetName+"/no1", 1, 9467);
    db.addSequence(2, datasetName+"/no2", 1, 8830);
    db.addSequence(3, datasetName+"/no3", 1, 11492);
    db.addSequence(4, datasetName+"/no4", 1, 10015);
    db.addSequence(5, datasetName+"/no5", 1, 12999);
    db.addSequence(6, datasetName+"/no6", 1, 11160);
    db.addSequence(7, datasetName+"/no7", 1, 9790);
    db.addSequence(8, datasetName+"/no8", 1, 10269);
    db.addSequence(9, datasetName+"/no9", 1, 11153);
    db.addSequence(10, datasetName+"/no10", 1, 10449);

    vector<int> trainS;
    trainS.push_back(3);
    trainS.push_back(4);
    db.setTrainSeqs(1, trainS);

    trainS.clear();
    trainS.push_back(1);
    trainS.push_back(7);
    db.setTrainSeqs(2, trainS);

    return db;
}

Dataset dataset_setup(string datasetName)
{
    if("Intel" == datasetName)
        return dataset_setup_Intel("Egocentric_Objects_Intel");
    else if("GTEA" == datasetName)
        return dataset_setup_GTEA("GTEA_plus");
    else if("Yale" == datasetName)
        return dataset_setup_Yale("Yale_Human_Grasp");
    else
    {
        cout << "dataset_setup: unrecognized dataset name <" << datasetName << ">\n";
        exit(1);
    }
}

vector<int> randsample(int n, int k)
{
    vector<int> output;
    set<int> tempSet;
	srand(1); //everytime calling randsample, produce the same number seqence
    while((int)output.size() < k)
    {
        int r = rand() % n;
        if(tempSet.find(r) == tempSet.end())
        {
            output.push_back(r);
            tempSet.insert(r);
        }
    }
    return output;
}

int getContourBig(Mat &prob, Mat &mask, double thres, vector<vector<Point> > &co, int &idx)
{
	Mat mb;
	medianBlur(prob, mb, 5);
	GaussianBlur(mb, mb, Size(9,9), 0, 0, BORDER_REFLECT);
    mask = (mb > thres)*255;
    if(0 == countNonZero(mask)) return -1;
	
    vector<Vec4i> hi;
    findContours(mask, co, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    mask *= 0;
    float maxval = -1.0;
    idx = -1;
    for(int c = 0; c < (int)co.size(); c++)
    {
        float area = contourArea(Mat(co[c]));
//        if(area < (mask.rows*mask.cols*0.01)) continue;	// too small
//        if(area > (mask.rows*mask.cols*0.9)) continue;	// too big
        if(area > maxval)
        {
            maxval = area;
            idx = c;
        }
    }

    if(idx!=-1) drawContours(mask, co, idx, Scalar::all(255.0), -1, CV_AA, hi, 1);
    else return -1;

    return 0;
}

/*******************************************/
vector<Action> Dataset::getActions(string seqName)
{
    char filename[500];
    filename[0] = 0;
    sprintf(filename, "%s/annotation/%s_action.txt", _dataDir.c_str(), seqName.c_str());
    return readActionAnnotations(filename);
}

vector<int> Dataset::getObjects(string seqName)
{
    char filename[500];
    filename[0] = 0;
    sprintf(filename, "%s/annotation/%s_label.txt", _dataDir.c_str(), seqName.c_str());
    return readObjectAnnotations(filename);
}

vector<Grasp> Dataset::getGrasps(string seqName)
{
    char filename[500];
    filename[0] = 0;
    sprintf(filename, "%s/annotation/%s_grasp.txt", _dataDir.c_str(), seqName.c_str());
    return readGraspAnnotations(filename);
}

vector<Grasp> Dataset::getfrequentGraspsInTraining(int version, int minFreq)
{
    vector<int> seqs = getTrainSeqs(version);
    map<string, int> graspCount;
    for(int i = 0; i < (int)seqs.size(); i++)
    {
        int seqNumber = seqs[i];
        string seqName = getSequence(seqNumber).seqName;
        vector<Grasp> grasps = getGrasps(seqName);
        for(int a = 0; a < (int)grasps.size(); a++)
        {
            if(graspCount.find(grasps[a].graspType) != graspCount.end())
                graspCount[grasps[a].graspType]++;
            else
                graspCount[grasps[a].graspType] = 1;
        }
    }
    vector<Grasp> output;
    for(map<string,int>::iterator it = graspCount.begin(); it != graspCount.end(); it++)
    {
        if(it->second >= minFreq)
        {
        	Grasp temp;
			temp.graspType = it->first;
            output.push_back(temp);
        }
    }
    return output;

}

vector<Grasp> Dataset::getfrequentGraspsInTesting(int version, int minFreq)
{
    vector<int> seqs = getTestSeqs(version);
    map<string, int> graspCount;
    for(int i = 0; i < (int)seqs.size(); i++)
    {
        int seqNumber = seqs[i];
        string seqName = getSequence(seqNumber).seqName;
        vector<Grasp> grasps = getGrasps(seqName);
        for(int a = 0; a < (int)grasps.size(); a++)
        {
            if(graspCount.find(grasps[a].graspType) != graspCount.end())
                graspCount[grasps[a].graspType]++;
            else
                graspCount[grasps[a].graspType] = 1;
        }
    }
    vector<Grasp> output;
    for(map<string,int>::iterator it = graspCount.begin(); it != graspCount.end(); it++)
    {
        if(it->second >= minFreq)
        {
            Grasp temp;
			temp.graspType = it->first;
            output.push_back(temp);
        }
    }
    return output;

}

vector<Grasp> Dataset::getFrequentGrasps(int version, int minFreqTraining, int minFreqTesting)
{
    vector<Grasp> graspsTraining = getfrequentGraspsInTraining(version, minFreqTraining);
    vector<Grasp> graspsTesting = getfrequentGraspsInTesting(version, minFreqTesting);
    vector<Grasp> freqGrasps;
    for(int g1 = 0; g1 < (int)graspsTraining.size(); g1++)
    {
        bool existInTest = false;
        for(int g2 = 0; g2 < (int)graspsTesting.size(); g2++)
        {
            if(graspsTraining[g1].graspType == graspsTesting[g2].graspType)
            {
                existInTest = true;
                break;
            }
        }
        if(existInTest)
            freqGrasps.push_back(graspsTraining[g1]);
    }

    return freqGrasps;

}

int Dataset::getGraspFreqInTraining(Grasp grasp, int version)
{
    vector<int> seqs = getTrainSeqs(version);
    int answer = 0;
    for(int i = 0; i < (int)seqs.size(); i++)
    {
        int seqNumber = seqs[i];
        string seqName = getSequence(seqNumber).seqName;
        vector<Grasp> grasps = getGrasps(seqName);
        for(int a = 0; a < (int)grasps.size(); a++)
            if(grasps[a].graspType == grasp.graspType)
                answer++;
    }
    return answer;

}

int Dataset::getGraspFreqInTesting(Grasp grasp, int version)
{
    vector<int> seqs = getTestSeqs(version);
    int answer = 0;
    for(int i = 0; i < (int)seqs.size(); i++)
    {
        int seqNumber = seqs[i];
        string seqName = getSequence(seqNumber).seqName;
        vector<Grasp> grasps = getGrasps(seqName);
        for(int a = 0; a < (int)grasps.size(); a++)
            if(grasps[a].graspType == grasp.graspType)
                answer++;
    }
    return answer;

}

vector<TrackInfo> Dataset::getTrackedHand(string seqName)
{
    char filename[500];
    filename[0] = 0;
    sprintf(filename, "%s/annotation/%s_track.txt", _dataDir.c_str(), seqName.c_str());
    return readTrackLog(filename);
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
        if(area > (dst.rows*dst.cols*0.4)) continue;	// too big
        areaRegion[-area] = c;
    }

    int count = (int)areaRegion.size();
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

    if(cfg.keyExists("version"))
    {
        _version = cfg.getValueOfKey<int>("version");
    }
    else
    {
        return ERR_CFG_VERSION_NONEXIST;
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
        for(size_t f = _startframe; f <= _endframe; f += _interval)
        {
            Mat anchorPoint(1, 10, CV_32FC1);
            stat = getHandRegion(_videoname[v], f , anchorPoint);
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
            Rect box = Rect(anchorPoint.at<float>(0,6), anchorPoint.at<float>(0,7), anchorPoint.at<float>(0,8),anchorPoint.at<float>(0,9));
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

int DataPreparation::getHandInfo(string seqName, int framenum, TrackInfo &handTrack, HandInfo &hInfo)
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
        cout << "[getHandInfo] failed to read hand image: " << ss.str() << endl;
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
        Rect box = boundingRect(contours[0]);
        if(handTrack.trackNum > 1)
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
        Rect box1 = boundingRect(contours[0]);
        RotatedRect rRect2 = fitEllipse(contours[1]);
        Rect box2 = boundingRect(contours[1]);

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
    return 0;
}

int DataPreparation::getGraspFromGTEA_old()
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
    Dataset db = dataset_setup("GTEA");
    for(int v = 0; v < (int)_videoname.size(); v++)
    {
        vector<Action> seqActions = db.getActions(_videoname[v]);
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

int DataPreparation::getGraspFromGTEA()
{
    int stat;
    stringstream ss;
    Dataset db = dataset_setup("GTEA");

    string datasetName = db.getDatasetName(); //name of base directory
    _handInfo = vector<vector<HandInfo> >(1);

    ss.str("");
    ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data/log.txt";
    ifstream infile(ss.str().c_str(), ios::in);
    if(infile)
    {
        infile.close();
        _handInfo[0] = readHandInfoLog(ss.str().c_str(), 0);
        cout << "finish reading data from log file\n";
        return 0;
    }
    infile.close();


    ss.str("");
    ss << "mkdir " + _rootname + "/grasp/" + datasetName + "/version" << _version;
    cout << ss.str() << endl;
    system(ss.str().c_str());

    vector<int> trainS = db.getTrainSeqs(_version);
    for(int s = 0;	s < (int)trainS.size(); s++)
    {
        string seqName = db.getSequence(trainS[s]).seqName;

        vector<TrackInfo> handTracks = db.getTrackedHand(seqName); // read hand tracking log

        for(int f = 0; f < (int)handTracks.size(); f++)
        {
            HandInfo hInfo;
            hInfo.seqNum = trainS[s];
            hInfo.frameNum = f+1;
            if(handTracks[f].trackNum == 0)
                continue;
            hInfo.objectId = 0;
            stat = getHandInfo(seqName, f+1, handTracks[f], hInfo);
            if(stat)
            {
                if(stat == ERR_CONTINUE)
                    continue;
                else
                    break;
            }

            // objectId begins from 1
            _handInfo[0].push_back(hInfo);
        }
        cout << "finish reading data from sequence: " << seqName << endl;
    }

    ss.str("");
    ss << "rm -r " + _rootname + "/grasp/" + datasetName + "/version" << _version << "/data";
    cout << ss.str() << endl;
    system(ss.str().c_str());

    ss.str("");
    ss << "mkdir " + _rootname + "/grasp/" + datasetName + "/version" << _version << "/data";
    cout << ss.str() << endl;
    system(ss.str().c_str());

    ss.str("");
    ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/log.txt";
    ofstream outfile(ss.str().c_str(), ios::out);
    for(int j = 0; j < (int)_handInfo[0].size(); j++)
    {
        ss.str("");
        ss << _handInfo[0][j].seqNum << "\t" << _handInfo[0][j].frameNum << "\t" << _handInfo[0][j].handState << "\t";
        ss << "(" << _handInfo[0][j].box[0].x << "," << _handInfo[0][j].box[0].y << "," << _handInfo[0][j].box[0].width << "," << _handInfo[0][j].box[0].height << ")\t";
        ss << "(" << _handInfo[0][j].box[1].x << "," << _handInfo[0][j].box[1].y << "," << _handInfo[0][j].box[1].width << "," << _handInfo[0][j].box[1].height << ")\n";
        outfile << ss.str();

        ss.str("");
        ss << _rootname + "/img/" + db.getSequence(_handInfo[0][j].seqNum).seqName << "/";
        ss << setw(8) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
        Mat img = imread(ss.str());
        if(!img.data)
        {
            cout << "failed to read color image: " << ss.str() << endl;
            return ERR_DATA_IMG_NONEXIST;
        }

        //save color image
        if(_handInfo[0][j].handState == HAND_L)
        {
            Mat img_roi(_handInfo[0][j].box[0].height, _handInfo[0][j].box[0].width, img.type());
            img(_handInfo[0][j].box[0]).copyTo(img_roi);

            ss.str("");
            ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/L_";
            ss << setw(2) << setfill('0') << _handInfo[0][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
            imwrite(ss.str(), img_roi);
        }
        else if(_handInfo[0][j].handState == HAND_R)
        {
            Mat img_roi(_handInfo[0][j].box[0].height, _handInfo[0][j].box[0].width, img.type());
            img(_handInfo[0][j].box[0]).copyTo(img_roi);

            ss.str("");
            ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/R_";
            ss << setw(2) << setfill('0') << _handInfo[0][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
            imwrite(ss.str(), img_roi);
        }
        else if(_handInfo[0][j].handState == HAND_LR)
        {
            Mat img_roi1(_handInfo[0][j].box[0].height, _handInfo[0][j].box[0].width, img.type());
            img(_handInfo[0][j].box[0]).copyTo(img_roi1);

            ss.str("");
            ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/L_";
            ss << setw(2) << setfill('0') << _handInfo[0][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
            imwrite(ss.str(), img_roi1);

            Mat img_roi2(_handInfo[0][j].box[1].height, _handInfo[0][j].box[1].width, img.type());
            img(_handInfo[0][j].box[1]).copyTo(img_roi2);

            ss.str("");
            ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/R_";
            ss << setw(2) << setfill('0') << _handInfo[0][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
            imwrite(ss.str(), img_roi2);
        }
        else if(_handInfo[0][j].handState == HAND_ITS)
        {
            Mat img_roi(_handInfo[0][j].box[0].height, _handInfo[0][j].box[0].width, img.type());
            img(_handInfo[0][j].box[0]).copyTo(img_roi);

            ss.str("");
            ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/data" << "/ITS";
            ss << setw(2) << setfill('0') << _handInfo[0][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[0][j].frameNum << ".jpg";
            imwrite(ss.str(), img_roi);
        }
        else
        {
            cout << "bad hand state at sequence " << _handInfo[0][j].seqNum << " frame " << _handInfo[0][j].frameNum << endl;
            continue;
        }

    }
    outfile.close();

    return 0;
}

int DataPreparation::getGraspFromIntel()
{
    int stat;
    stringstream ss;
    Dataset db = dataset_setup("Intel");

    string datasetName = db.getDatasetName(); //name of base directory

    _handInfo = vector<vector<HandInfo> >(INTEL_OBJECT_SIZE);
    bool isExist = true;
    for(int i = 0; i < INTEL_OBJECT_SIZE; i++)
    {
        ss.str("");
        ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/log.txt";
        ifstream infile(ss.str().c_str(), ios::in);
        if(!infile)
        {
            isExist = false;
            break;
        }
        else
        {
            infile.close();
            _handInfo[i] = readHandInfoLog(ss.str().c_str(), i+1);
        }
    }

    if(isExist)
    {
        cout << "finish reading data from log file\n";
        return 0;
    }

    ss.str("");
    ss << "mkdir " + _rootname + "/grasp/" + datasetName + "/version" << _version;
    cout << ss.str() << endl;
    system(ss.str().c_str());

    vector<int> trainS = db.getTrainSeqs(_version);
    _handInfo = vector<vector<HandInfo> >(INTEL_OBJECT_SIZE);
    for(int s = 0;  s < (int)trainS.size(); s++)
    {
        string seqName = db.getSequence(trainS[s]).seqName;
        vector<int> objectList = db.getObjects(seqName); // read object label

        vector<TrackInfo> handTracks = db.getTrackedHand(seqName); // read hand tracking log

        for(int f = 0; f < (int)objectList.size(); f++)
        {
            HandInfo hInfo;
            hInfo.seqNum = trainS[s];
            hInfo.frameNum = f+1;
            if(objectList[f] == 0)
                continue;
            hInfo.objectId = objectList[f];
            stat = getHandInfo(seqName, f+1, handTracks[f], hInfo);
            if(stat)
            {
                if(stat == ERR_CONTINUE)
                    continue;
                else
                    break;
            }

            // objectId begins from 1
            _handInfo[hInfo.objectId-1].push_back(hInfo);
        }
        cout << "finish reading data from sequence: " << seqName << endl;
    }

    for(int i = 0; i < INTEL_OBJECT_SIZE; i++)
    {
        ss.str("");
        ss << "rm -r " + _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1;
        cout << ss.str() << endl;
        system(ss.str().c_str());

        ss.str("");
        ss << "mkdir " + _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1;
        cout << ss.str() << endl;
        system(ss.str().c_str());

        ss.str("");
        ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/log.txt";
        ofstream outfile(ss.str().c_str(), ios::out);
        for(int j = 0; j < (int)_handInfo[i].size(); j++)
        {
            ss.str("");
            ss << _handInfo[i][j].seqNum << "\t" << _handInfo[i][j].frameNum << "\t" << _handInfo[i][j].handState << "\t";
            ss << "(" << _handInfo[i][j].box[0].x << "," << _handInfo[i][j].box[0].y << "," << _handInfo[i][j].box[0].width << "," << _handInfo[i][j].box[0].height << ")\t";
            ss << "(" << _handInfo[i][j].box[1].x << "," << _handInfo[i][j].box[1].y << "," << _handInfo[i][j].box[1].width << "," << _handInfo[i][j].box[1].height << ")\n";
            outfile << ss.str();

            ss.str("");
            ss << _rootname + "/img/" + db.getSequence(_handInfo[i][j].seqNum).seqName << "/";
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
                ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/L_";
                ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
                imwrite(ss.str(), img_roi);
            }
            else if(_handInfo[i][j].handState == HAND_R)
            {
                Mat img_roi(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
                img(_handInfo[i][j].box[0]).copyTo(img_roi);

                ss.str("");
                ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/R_";
                ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
                imwrite(ss.str(), img_roi);
            }
            else if(_handInfo[i][j].handState == HAND_LR)
            {
                Mat img_roi1(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
                img(_handInfo[i][j].box[0]).copyTo(img_roi1);

                ss.str("");
                ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/L_";
                ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
                imwrite(ss.str(), img_roi1);

                Mat img_roi2(_handInfo[i][j].box[1].height, _handInfo[i][j].box[1].width, img.type());
                img(_handInfo[i][j].box[1]).copyTo(img_roi2);

                ss.str("");
                ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/R_";
                ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
                imwrite(ss.str(), img_roi2);
            }
            else if(_handInfo[i][j].handState == HAND_ITS)
            {
                Mat img_roi(_handInfo[i][j].box[0].height, _handInfo[i][j].box[0].width, img.type());
                img(_handInfo[i][j].box[0]).copyTo(img_roi);

                ss.str("");
                ss << _rootname + "/grasp/" + datasetName + "/version" << _version << "/object-" << i+1 << "/ITS";
                ss << setw(2) << setfill('0') << _handInfo[i][j].seqNum << "_" << setw(6) << setfill('0') << _handInfo[i][j].frameNum << ".jpg";
                imwrite(ss.str(), img_roi);
            }
            else
            {
                cout << "bad hand state at sequence " << _handInfo[i][j].seqNum << " frame " << _handInfo[i][j].frameNum << endl;
                continue;
            }

        }
        outfile.close();
    }

    return 0;
}

