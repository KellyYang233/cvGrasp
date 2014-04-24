/*
 *  cvGrasp.cpp
 *  Discovering grasp types using computer vision
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "cvGrasp.hpp"

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

int cvGrasp::procIntelGrasp()
{
	int stat = 0;

	stat = _dp.initialize(_cfg);
	if(stat) return stat;

	stat = _dp.getGraspFromIntel();
	if(stat) return stat; 
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
	else if("intel" == operation)
	{
		stat = grasp.procIntelGrasp();
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
		return -1;
	}
	else
	{
		cout << "failed with error: <" << stat << ">\n";
		return -1;
	}

}
