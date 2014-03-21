/*
 *  Skeleton.cpp
 *  class for skeleton extraction
 *
 *  Created by Minjie Cai on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Skeleton.hpp"

#define PI 3.142
#define D2_THRESHOLD 4.0

Skeleton::Skeleton(Mat &prob, Mat &img)
{
	m_prob = prob;
	m_img = img;
	m_isSkeleton = true;
}

int Skeleton::getContourBig(Mat src, Mat &dst, double thres, vector<vector<Point> > &co, int &idx)
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

//relevance measurement of contribution to the shape
//K(S1,S2) = angle(S1,S2)length(S1)length(S2)/(length(S1)+length(S2))
double Skeleton::getRelevance(vector<Point> contour, int index, double perimeter)
{
	//set neighboring point
	int i0 = index - 1;
	int i1 = index;
	int i2 = index + 1;

	int m = (int)contour.size();
	if(i0 < 0) i0 = m-1;
	else if(i2 > m-1) i2 = 0;

	int seg1x = contour[i1].x - contour[i0].x;
	int seg1y = contour[i1].y - contour[i0].y;
	int seg2x = contour[i1].x - contour[i2].x;
	int seg2y = contour[i1].y - contour[i2].y;

	double l1 = sqrt(seg1x*seg1x + seg1y*seg1y);
	double l2 = sqrt(seg2x*seg2x + seg2y*seg2y);

	//turning angle (0-180)
	double angle = 180 - acos((seg1x*seg2x + seg1y*seg2y)/l1/l2) * 180 / PI;

	//relevance measure
	double K = angle*l1*l2 / (l1+l2);

	//normalize
	K /= perimeter;
	return K;
}

bool Skeleton::isBlocked(vector<Point> contour, int index)
{
	bool stat = false;

	//set neighboring point
	int i0 = index - 1;
	int i1 = index;
	int i2 = index + 1;

	int m = (int)contour.size();
	if(i0 < 0) i0 = m-1;
	else if(i2 > m-1) i2 = 0;

	//bounding box
	int minx = min(contour[i0].x, min(contour[i1].x, contour[i2].x));
	int miny = min(contour[i0].y, min(contour[i1].y, contour[i2].y));
	int maxx = max(contour[i0].x, max(contour[i1].x, contour[i2].x));
	int maxy = max(contour[i0].y, max(contour[i1].y, contour[i2].y));

	//make up a list exclusive of point i0,i1,i2
	vector<int> list(0);
	for(int i = 0; i < m; i++)
	{
		if(i != i0 && i != i1 && i != i2)
			list.push_back(i);
	}

	for(int i = 0; i < (int)list.size(); i++)
	{
		int px = contour[i].x;
		int py = contour[i].y;
		if(px >= minx && px <= maxx && py >= miny && py <= maxy)
		{
			Point a(contour[i1].x-contour[i0].x, contour[i1].y-contour[i0].y);
			Point b(contour[i2].x-contour[i1].x, contour[i2].y-contour[i1].y);
			Point c(contour[i0].x-contour[i2].x, contour[i0].y-contour[i2].y);

			Point e0(contour[i0].x-px, contour[i0].y-py);
			Point e1(contour[i1].x-px, contour[i1].y-py);
			Point e2(contour[i2].x-px, contour[i2].y-py);

			//value of cross-product
			int d0 = a.x*e0.y - a.y*e0.x;
			int d1 = b.x*e1.y - b.y*e1.x;
			int d2 = c.x*e2.y - c.y*e2.x;

			//inside triangle <a,b,c> ?
			stat = ((d0>0) && (d1>0) && (d2>0)) || ((d0<0) && (d1<0) && (d2<0));
			if(stat) break;
		}
	}

	return stat;
}

void Skeleton::curveEvolution(vector<Point> &contourDCE, const double maxValue, const int minNum)
{
	bool stat;
	double perimeter = arcLength(contourDCE, true);
	vector<float> relevance((int)contourDCE.size(), 0);

	//get relevance measure for every contour point
	for(int i = 0; i < (int)contourDCE.size(); i++)
	{
		relevance[i] = getRelevance(contourDCE, i, perimeter);
	}

	//iteratively remove one contour point with smallest relevance measure
	while(1)
	{
		if(minNum >= (int)contourDCE.size())
		{
		//	cout << "minimum contour vertex num: " << (int)contourDCE.size() << endl;
			break;
		}

		float minVal = FLT_MAX;
		int index = -1;
		for(int j = 0; j < (int)relevance.size(); j++)
		{
			if(relevance[j] < minVal)
			{
				minVal = relevance[j];
				index = j;
			}
		}
		if(index == -1)
		{
			cout << "bad index of relevance measure\n";
			return;
		}
		else if(relevance[index] >= maxValue)
		{
		//	cout << "minimum relevance value: " << relevance[index] << endl;
			break;
		}

		//test blocking
		stat = isBlocked(contourDCE, index);
		if(stat)
		{
			Mat label;
			Mat src((int)relevance.size(), 1, CV_32FC1);
			for(int j = 1; j < (int)relevance.size(); j++)
				src.at<float>(j,0) = relevance[j];
			sortIdx(src, label, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			minVal = FLT_MAX;
			for(int j = 1; j < label.rows; j++)
			{
				index = label.at<int>(j,0);
				stat = isBlocked(contourDCE, index);
				if(false == stat)
				{
					minVal = relevance[index];
					break;
				}
			}

			if(minVal > maxValue) break;
		}

		//remove vertex
		contourDCE.erase(contourDCE.begin() + index);
		relevance.erase(relevance.begin() + index);

		//neighbouring vertices
		int i0 = index - 1;
		int i1 = index;
		if(i0 < 0)	i0 = (int)contourDCE.size() - 1;
		else if(i1 > (int)contourDCE.size() - 1) i1 = 0;

		//update relevance measure for neighbouring vertices
		relevance[i0] = getRelevance(contourDCE, i0, perimeter);
		relevance[i1] = getRelevance(contourDCE, i1, perimeter);
	}
}

void Skeleton::findConvex(vector<Point> list, vector<Point> &convex, vector<Point> &concave)
{
	vector<int> cross_product((int)list.size(), 0);
	int index = 0;
	int dx1 = list[index].x - (*list.end()).x;
	int dy1 = list[index].y - (*list.end()).y;
	int dx2 = list[index+1].x - list[index].x;
	int dy2 = list[index+1].y - list[index].y;
	cross_product[index] = dx1*dy2 - dy1*dx2;

	for(index = 1; index < (int)list.size()-1; index++)
	{
		dx1 = list[index].x - list[index-1].x;
		dy1 = list[index].y - list[index-1].y;
		dx2 = list[index+1].x - list[index].x;
		dy2 = list[index+1].y - list[index].y;
		cross_product[index] = dx1*dy2 - dy1*dx2;
	}

	index = (int)list.size() - 1;
	dx1 = list[index].x - list[index-1].x;
	dy1 = list[index].y - list[index-1].y;
	dx2 = (*list.begin()).x - list[index].x;
	dy2 = (*list.begin()).y - list[index].y;
	cross_product[index] = dx1*dy2 - dy1*dx2;

	//count and seperate opposite direction of cross product
	int num1 = 0, num2 = 0;
	vector<Point> p1(0);
	vector<Point> p2(0);
	for(int i = 0; i < (int)cross_product.size(); i++)
	{
		if(cross_product[i] > 0)
		{
			p1.push_back(list[i]);
			num1++;
		}
		else
		{
			p2.push_back(list[i]);
			num2++;
		}
	}

	//num of convex is bigger than num of concave in polygon
	if(num1 >= num2)
	{
		convex = p1;
		concave = p2;
	}
	else
	{
		convex = p2;
		concave = p1;
	}
}

void Skeleton::prunConvex(vector<Point> &convex, vector<Point> concave, const double thres, vector<Point> contour)
{
	vector<double> minDist((int)convex.size(), DBL_MAX);
	for(int i = 0; i < (int)concave.size(); i++)
	{
		for(int j = 0; j < (int)convex.size(); j++)
		{
			double D2 = pow(convex[j].x-concave[i].x, 2) + pow(convex[j].y-concave[i].y, 2);
			if(D2 < minDist[j]) minDist[j] = D2;
		}
	}
	//prun convex by distance to nearest concave
	for(int i = (int)convex.size()-1; i >= 0; i--)
	{
		if(minDist[i] < thres*thres) convex.erase(convex.begin()+i);
	}

	//prun convex by finger shape heuristic
	Rect box = boundingRect(contour);
	for(int i = (int)convex.size()-1; i >= 0; i--)
	{
		int idx = -1, idx1 = 0, idx2 = 0;
		for(int j = 0; j < (int)contour.size(); j++)
		{
			if(contour[j].x == convex[i].x && contour[j].y == convex[i].y) idx = j;
		}
		if(idx == -1) {cout << "ERROR: prun convex\n"; return;}
		idx1 = idx-15>=0? idx-15 : idx+(int)contour.size()-15;
		idx2 = idx+15<(int)contour.size()? idx+15 : idx-(int)contour.size()+15;
		Point v1 = Point(contour[idx1].x-contour[idx].x, contour[idx1].y-contour[idx].y);
		Point v2 = Point(contour[idx2].x-contour[idx].x, contour[idx2].y-contour[idx].y);
		double dot_p = (v1.x*v2.x + v1.y*v2.y) / sqrt((v1.x*v1.x+v1.y*v1.y)*(v2.x*v2.x+v2.y*v2.y));
		if(dot_p < 0)
		{
			convex.erase(convex.begin()+i);
			continue;
		}
		if(convex[i].y > box.y+box.height-20 || convex[i].x > box.x+box.width-20)
		{
			convex.erase(convex.begin()+i);
		}
	}
}

void Skeleton::markContourPartition(vector<Point> contour, vector<Point> convex, Mat &mark)
{
	int temp1 = 0, temp2 = 0, first = 0;
	for(int i = 0; i < (int)convex.size()-1; i++)
	{
		for(int j = 0; j < (int)contour.size(); j++)
		{
			if(contour[j].x == (int)convex[i].x && contour[j].y == (int)convex[i].y)
				temp1 = j;
			if(contour[j].x == (int)convex[i+1].x && contour[j].y == (int)convex[i+1].y)
				temp2 = j;
		}
		if(0 == i) first = temp1;
//		cout << "<" << i << ">[" << temp1 << ", " << temp2 << "]\n";
		if(temp1 < temp2)
		{
			for(int j = temp1; j < temp2; j++)
				mark.at<int>(contour[j]) = i + 1;
		}
		else if(temp1 > temp2)
		{
			for(int j = temp1; j < (int)contour.size(); j++)
				mark.at<int>(contour[j]) = i + 1;
			for(int j = 0; j < temp2; j++)
				mark.at<int>(contour[j]) = i + 1;
		}
		else
		{
			cout << "ERROR: invalid convex vertex\n";
		}
		temp1 = temp2;
	}

	temp2 = first;
//	cout << "<last>[" << temp1 << ", " << temp2 << "]\n";
	if(temp1 < temp2)
	{
		for(int j = temp1; j < temp2; j++)
			mark.at<int>(contour[j]) = 0;
	}
	else if(temp1 > temp2)
	{
		for(int j = temp1; j < (int)contour.size(); j++)
			mark.at<int>(contour[j]) = 0;
		for(int j = 0; j < temp2; j++)
			mark.at<int>(contour[j]) = 0;
	}
	else
	{
		cout << "ERROR: invalid convex vertex <last index>\n";
	}
}

void Skeleton::checkSkeletonPoint(Point center)
{
	//termination check
	if(center.x < 0 || center.x > m_dt.cols-1 || center.y < 0 || center.y > m_dt.rows-1) return;
	if(m_dt.at<float>(center) == 0) return;

	int mark1 = 0, mark2 = 0;
	Point nearest;
	//8-connectivity neighborhood
	const Point neighbor[] =
	{
		Point(center.x-1, center.y-1),
		Point(center.x, center.y-1),
		Point(center.x+1, center.y-1),
		Point(center.x-1, center.y),
		Point(center.x+1, center.y),
		Point(center.x-1, center.y+1),
		Point(center.x, center.y+1),
		Point(center.x+1, center.y+1)
	};

	nearest.x = m_label_contour[m_labels.at<int>(center)].x;
	nearest.y = m_label_contour[m_labels.at<int>(center)].y;
	if(nearest.x == 0 && nearest.y == 0) return; //can't find corresponding label from contour point
	mark1 = m_mark.at<int>(nearest);
	Point Q0(nearest.x-center.x, nearest.y-center.y);
	//if any pair of contour points (Q0,Q1) satisfies the connectivity criterion, pixel 'center' is a skeleton point
	for(int i = 0; i < 8; i++)
	{
		if(neighbor[i].x < 0 || neighbor[i].x > m_dt.cols-1 || neighbor[i].y < 0 || neighbor[i].y > m_dt.rows-1) continue;
		if(m_dt.at<float>(neighbor[i]) == 0) continue;
		nearest.x = m_label_contour[m_labels.at<int>(neighbor[i])].x;
		nearest.y = m_label_contour[m_labels.at<int>(neighbor[i])].y;
		if(nearest.x == 0 && nearest.y == 0) continue;
		mark2 = m_mark.at<int>(nearest);
		if(mark1 == mark2) continue; //for skeleton pruning
		Point Q1(nearest.x-center.x, nearest.y-center.y);
		double D2 = pow(Q1.x-Q0.x, 2) + pow(Q1.y-Q0.y, 2);
		int maxXY = max(abs(Q1.x-Q0.x), abs(Q1.y-Q0.y));
		int diff = pow(Q1.x, 2) + pow(Q1.y, 2) - pow(Q0.x, 2) - pow(Q0.y, 2);
//		double pro = pow(dt.at<float>(center), 2);
		//connectivity criterion
		if(D2 >= D2_THRESHOLD/*std::min(pro, D2_THRESHOLD)*/ && maxXY >= diff)
		{
			m_skeleton.at<uchar>(center) = 255;
			break;
		}
	}

	if(255 == m_skeleton.at<uchar>(center))
	{
		for(int j = 0; j < 8; j++)
		{
			if(m_skeleton.at<uchar>(neighbor[j]) == 0)
				checkSkeletonPoint(neighbor[j]);
		}
	}
	else
	{
		m_skeleton.at<uchar>(center) = 1;
	}

	return;
}

void Skeleton::findPath(vector<Point> endpoints, const int index, vector<vector<Point> > &endpaths)
{
	Point src(endpoints[index].x, endpoints[index].y);
	for(int i = 0; i < (int)endpoints.size()-1; i++)
	{
		int idx = (index+i+1) % (int)endpoints.size();//start from endpoint that is next to the current endpoint
		Point dst(endpoints[idx].x, endpoints[idx].y);
		//cout << "to find path between [" << src.x << ", " << src.y << "] and [" << dst.x << ", " << dst.y << "]\n";
		Mat visited = Mat::zeros(m_skeleton.size(), CV_8UC1);
		stack<Point> accessStack;
		vector<Point> path;
		accessStack.push(src);
		visited.at<uchar>(src) = 1;

		//depth first search
		while((int)accessStack.size() > 0)
		{
			Point cur(accessStack.top());
			path.push_back(cur);
			if(cur.x == dst.x && cur.y == dst.y) break;
			accessStack.pop();
			for(int row = cur.y-1; row <= cur.y+1; row++)
				for(int col = cur.x-1; col <= cur.x+1; col++)
				{
					if(row < 0 || row > m_skeleton.rows-1 || col < 0 || col > m_skeleton.cols-1) continue;
					if(0 == visited.at<uchar>(row,col) && 255 == m_skeleton.at<uchar>(row,col))
					{
						visited.at<uchar>(row,col) = 1;
						accessStack.push(Point(col,row));
					}
				}
		}

		for(int j = (int)path.size()-1; j > 1; j--)
		{
			Point former(path[j]);
			Point latter(path[j-1]);
			if(pow(former.x-latter.x,2) + pow(former.y-latter.y,2) > 2)
				path.erase(path.begin()+j-1);
		}

		endpaths[i] = path;
	}
}

int Skeleton::findSkeleton(double segmentThres, double dceCrit, int dceNum)
{
	int stat = 0;
	int idx;
        vector<vector<Point> > co;
	// choose search window for palm center and fingertip location
	Mat binary(m_prob.size(), CV_8UC1);
	stat = getContourBig(m_prob, binary, segmentThres, co, idx);
	if(stat)
	{
		cout << "failed to get contour\n";
	}

	// distance transform at search window using second variant of <distanceTransform>
	m_labels = Mat::zeros(m_dt.size(), CV_32SC1);
	double minVal, maxVal;
	Point minLoc, maxLoc;
	distanceTransform(binary, m_dt, m_labels, CV_DIST_L2, CV_DIST_MASK_5, DIST_LABEL_PIXEL);
	minMaxLoc(m_dt, &minVal, &maxVal, NULL, &maxLoc);

	// contour partitioning by Discrete Curve Evolution
	m_contourDCE = m_contour = co[idx];
	curveEvolution(m_contourDCE, dceCrit, dceNum);//Discrete Curve Evolution
	findConvex(m_contourDCE, m_convex, m_concave); //find convex polygon vertex
	prunConvex(m_convex, m_concave, 15, co[idx]);//prun convex vertex with small distance to nearest concave vertex
	if((int)m_convex.size() < 2)
	{
		m_isSkeleton = false;
		return -1;
	}
	m_mark = Mat::zeros(m_dt.size(), CV_32SC1);
	markContourPartition(m_contour, m_convex, m_mark);//mark contour partition for skeleton prunning

	minMaxLoc(m_labels, NULL, &maxVal, NULL, NULL);
	//vector storing Voronoi label of contour points
	m_label_contour.assign((int)maxVal, Point(0,0));
	for(int i = 0; i < (int)m_contour.size(); i++)
	{
		int label = m_labels.at<int>(m_contour[i]);
		m_label_contour[label] = m_contour[i];
	}

	//8. extract skeleton based on connectivity criterion and contour partition
	m_skeleton = Mat::zeros(m_dt.size(), CV_8UC1);
	m_skeleton.at<uchar>(maxLoc) = 255;
	checkSkeletonPoint(maxLoc);

	return 0;
}

void Skeleton::calcSkeleton()
{
	if(!m_isSkeleton) return;
	// transfer skeleton map to node list
	vector<Point> ends(0);
	vector<Point> joints(0);
	for(int row = 0; row < m_skeleton.rows; row++)
		for(int col = 0; col < m_skeleton.cols; col++)
		{
			if(255 == m_skeleton.at<uchar>(row, col))
			{
				int cnt = 0;
				for(int k = row-1; k <= row+1; k++)
					for(int l = col-1; l <= col+1; l++)
					{
						if(k == row && l == col) continue;
						if(k < 0 || k > m_skeleton.rows-1 || l < 0 || l > m_skeleton.cols-1) continue;
						if(255 == m_skeleton.at<uchar>(k, l))
							cnt++;
					}
				if(cnt == 1 || cnt == 2)
					ends.push_back(Point(col, row));
				else if(cnt > 2)
					joints.push_back(Point(col, row));
			}
		}

	// set order for endpoints along the contour
	for(int i = 0; i < (int)m_convex.size(); i++)
	{
		int max_diff = 200;
		int index = -1;
		for(int j = 0; j < (int)ends.size(); j++)
		{
			int diff = pow(m_convex[i].x-ends[j].x, 2) + pow(m_convex[i].y-ends[j].y, 2);
			if(diff < max_diff)
			{
				max_diff = diff;
				index = j;
			}
		}
		if(index != -1) m_endpoints.push_back(ends[index]);
	}
	if((int)m_endpoints.size() < 2)
	{
		m_isSkeleton = false;
		return;
	}

	//visualize skeleton and endpoints
	for(int row = 0; row < m_skeleton.rows; row++)
                for(int col = 0; col < m_skeleton.cols; col++)
                {
                        if(255 == m_skeleton.at<uchar>(row, col))
                                circle(m_img, Point(col, row), 1, CV_RGB(0,255,0), -1);
                }
        for(int i = 0; i < (int)m_convex.size(); i++)
                circle(m_img, m_convex[i], 4, CV_RGB(255,0,0), 1);
        for(int i = 0; i < (int)m_endpoints.size(); i++)
                circle(m_img, m_endpoints[i], 2, CV_RGB(0,0,255), -1);

	//average of distance transform
	float sum_dt = 0;
	int num_nonzero = 0;
	for(int i = 0; i < (int)m_dt.rows; i++)
		for(int j = 0; j < (int)m_dt.cols; j++)
		{
			if(m_dt.at<float>(i,j) > 0)
			{
				sum_dt += m_dt.at<float>(i,j);
				num_nonzero++;
			}
		}
	float dt_norm = sum_dt/num_nonzero;

	int num_end = (int)m_endpoints.size();
	vector<vector<vector<Point> > > allPaths;
	//find path to every other endpoints for each endpoint
	for(int i = 0; i < num_end; i++)
	{
		vector<vector<Point> > endpaths(num_end-1);
		findPath(m_endpoints, i, endpaths);
		allPaths.push_back(endpaths);

		vector<vector<float> > rad(num_end-1);
		for(int j = 0; j < num_end-1; j++)
		{
			int len = (int)endpaths[j].size();
			int num_sample = 8;
			int step = floor(len/(num_sample+1));
			for(int k = 1; k <= num_sample; k++)
			{
				float rad_norm = m_dt.at<float>(endpaths[j][k*step]) / dt_norm;//radium normalization
				rad[j].push_back(rad_norm);
			}
		}
		m_radii.push_back(rad);
	}

	int sum_len = 0;
	for(int i = 0; i < num_end; i++)
		for(int j = 0; j < num_end - 1; j++)
			sum_len += (int)allPaths[i][j].size();
	float len_norm = sum_len*1.0 / num_end / (num_end-1); //average path length
	for(int i = 0; i < num_end; i++)
	{
		vector<float> len_path(num_end-1);
		for(int j = 0; j < num_end-1; j++)
			len_path[j] = (int)allPaths[i][j].size() / len_norm; //path length normalization
		m_length.push_back(len_path);
	}
}




