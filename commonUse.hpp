/*
 *  commonUse.hpp
 *  CommonUse
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEFINE_COMMONUSE_HPP
#define DEFINE_COMMONUSE_HPP

#include <string>
#include <vector>
#include <stack>
#include <list>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <algorithm>

using namespace std;
#define PI 3.141593
#define MAX_LOOP 100000

enum error_t
{
	ERR_CLI_PARAM = 1000,
	ERR_CLI_PARAM_INVALID,
	ERR_CLI_VALUE_INVALID,
	ERR_CFG_PARAM = 1100,
	ERR_CFG_ROOTNAME_NONEXIST,
	ERR_CFG_VIDEONAME_NONEXIST,
	ERR_CFG_STARTFRAME_NONEXIST,
	ERR_CFG_ENDFRAME_NONEXIST,
	ERR_CFG_INTERVAL_NONEXIST,
	ERR_CFG_FEATURE_NONEXIST,
	ERR_CFG_CLUSTER_NONEXIST,
	ERR_CFG_THRESHOLD_NONEXIST,
	ERR_CFG_VALUE = 1150,
	ERR_CFG_FRAME_INVALID,
	ERR_CFG_FEATURE_INVALID,
	ERR_CFG_CLUSTER_INVALID,
	ERR_DATA = 1200,
	ERR_DATA_HAND_NONEXIST,
	ERR_DATA_IMG_NONEXIST,
	ERR_FEATURE = 1300,
	ERR_FEATURE_DETECTOR_CREAT,
	ERR_FEATURE_EXTRACTOR_CREAT,
	ERR_FEATURE_IMG_NONEXIST,
	ERR_FEATURE_HAND_NONEXIST,
	ERR_FEATURE_FILE_OPEN,
	ERR_CLUSTER = 1400,
	ERR_CLUSTER_FEATURE_NONEXIST,
	ERR_CLUSTER_FEATURE_DIMENSION,
	ERR_CLUSTER_HTML_OPEN
};

template <typename T>
void print_matrix(const vector<vector<T> >& C, int rows, int cols)
{
        int i,j;
        cout << endl;
        for(i=0; i<rows; i++)
        {
                cout << " [";
                for(j=0; j<cols; j++)
                {
                        cout << C[i][j] << " ";
                }
                cout << "]\n";

        }
        cout << endl;
}

inline void exitWithError(const string &error) 
{
	cout << error;
	cin.ignore();
	cin.get();

	exit(EXIT_FAILURE);
}

class Convert
{
public:
	template <typename T>
	static string T_to_string(T const &val) 
	{
		ostringstream ostr;
		ostr << val;

		return ostr.str();
	}
		
	template <typename T>
	static T string_to_T(string const &val) 
	{
		istringstream istr(val);
		T returnVal;
		if (!(istr >> returnVal))
			exitWithError("CFG: Not a valid type received!\n");

		return returnVal;
	}
private:
};

class ConfigFile
{
private:
	map<string, string> contents;
	string fName;

	void removeComment(string &line) const
	{
		if (line.find('#') != line.npos)
			line.erase(line.find('#'));
	}

	bool onlyWhitespace(const string &line) const
	{
		return (line.find_first_not_of(' ') == line.npos);
	}
	bool validLine(const string &line) const
	{
		string temp = line;
		temp.erase(0, temp.find_first_not_of("\t "));
		if (temp[0] == '=')
			return false;

		for (size_t i = temp.find('=') + 1; i < temp.length(); i++)
			if (temp[i] != ' ')
				return true;

		return false;
	}

	void extractKey(string &key, size_t const &sepPos, const string &line) const
	{
		key = line.substr(0, sepPos);
		if (key.find('\t') != line.npos || key.find(' ') != line.npos)
			key.erase(key.find_first_of("\t "));
	}
	void extractValue(string &value, size_t const &sepPos, const string &line) const
	{
		value = line.substr(sepPos + 1);
		value.erase(0, value.find_first_not_of("\t "));
		value.erase(value.find_last_not_of("\t ") + 1);
	}

	void extractContents(const string &line) 
	{
		string temp = line;
		temp.erase(0, temp.find_first_not_of("\t "));
		size_t sepPos = temp.find('=');

		string key, value;
		extractKey(key, sepPos, temp);
		extractValue(value, sepPos, temp);

		if (!keyExists(key))
			contents.insert(pair<string, string>(key, value));
		else
			exitWithError("CFG: Can only have unique key names!\n");
	}

	void parseLine(const string &line, size_t const lineNo)
	{
		if (line.find('=') == line.npos)
			exitWithError("CFG: Couldn't find separator on line: " + Convert::T_to_string(lineNo) + "\n");

		if (!validLine(line))
			exitWithError("CFG: Bad format for line: " + Convert::T_to_string(lineNo) + "\n");

		extractContents(line);
	}

	void ExtractKeys()
	{
		ifstream file;
		file.open(fName.c_str());
		if (!file)
			exitWithError("CFG: File " + fName + " couldn't be found!\n");

		string line;
		size_t lineNo = 0;
		while (getline(file, line))
		{
			lineNo++;
			string temp = line;

			if (temp.empty())
				continue;

			removeComment(temp);
			if (onlyWhitespace(temp))
				continue;

			parseLine(temp, lineNo);
		}

		file.close();
	}
public:
	ConfigFile() {}
	ConfigFile(const string &fName)
	{
		this->fName = fName;
		ExtractKeys();
	}

	bool keyExists(const string &key) const
	{
		return contents.find(key) != contents.end();
	}

	template <typename ValueType>
	ValueType getValueOfKey(const string &key, ValueType const &defaultValue = ValueType()) const
	{
		if (!keyExists(key))
			return defaultValue;

		return Convert::string_to_T<ValueType>(contents.find(key)->second);
	}
};

typedef enum
{
	HUNGARIAN_MODE_MINIMIZE_COST,
	HUNGARIAN_MODE_MAXIMIZE_UTIL
}MODE;

typedef enum
{
        HUNGARIAN_NOT_ASSIGNED,
        HUNGARIAN_ASSIGNED
}ASSIGN;

class Assignment
{
public:
	vector<vector<int> > org_cost, cost;          //cost matrix
	int n, max_match, penalty;        //n workers and n jobs
	vector<int> lx, ly;        //labels of X and Y parts
	vector<int> xy;               //xy[x] - vertex that is matched with x,
	vector<int> yx;               //yx[y] - vertex that is matched with y
	vector<bool> S, T;         //sets S and T in algorithm
	vector<int> slack;            //as in the algorithm description
	vector<int> slackx;           //slackx[y] such a vertex, that l(slackx[y]) + l(y) - w(slackx[y],y) = slack[y]
	vector<int> prev;             //array for memorizing alternating paths

	Assignment();
        Assignment(const vector<vector<int> >&, int, int, MODE);
	void init_labels();
	void augment();                //main function of the algorithm

private:
	void update_labels();
	void add_to_tree(int x, int prevx);
};

// Graph is represented using adjacency list. Every node of adjacency list
// contains vertex number of the vertex to which edge connects. It also
// contains weight of the edge
class AdjListNode
{
	int v;
	float weight;
public:
	AdjListNode(int _v, float _w)
	{
		v = _v;
		weight = _w;
	}
	int getV()
	{
		return v;
	}
	float getWeight()
	{
		return weight;
	}
};

// Class to represent a graph using adjacency list representation
class Graph
{
	int V;    // No. of vertices'

	// Pointer to an array containing adjacency lists
	list<AdjListNode> *adj;

	// A function used by shortestPath
	void topologicalSortUtil(int v, bool visited[], stack<int> &Stack);
public:
	Graph(int V);   // Constructor

	// function to add an edge to graph
	void addEdge(int u, int v, float weight);

	// Finds shortest paths from given source vertex
	void shortestPath(int s, int d, float &minDist);
};

#endif
