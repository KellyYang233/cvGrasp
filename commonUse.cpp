/** commonUse.cpp
 *  CommonUse
 *
 *  Created by Minjie Cai on 10/17/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 **/

#include "commonUse.hpp"

Assignment::Assignment(const vector<vector<int> >& input_matrix, int rows, int cols, MODE mode)
{       
        int max_cost = 0;             
        int org_cols = cols;
        int org_rows = rows;
        
        // is the matrix square?
        // if no, expand with 0-cols / 0-cols
        
        if(rows!=cols)
        {
                rows = std::max(cols, rows);
                cols = rows;
        }
        
	//member variable initialize
        n = rows;
	max_match = 0;
	org_cost.resize(n, vector<int>(n, 0));
	cost.resize(n, vector<int>(n, 0));
	lx.resize(n, 0);
	ly.resize(n, 0);
	xy.resize(n, -1);
	yx.resize(n, -1);
	S.resize(n, false);
	T.resize(n, false);
	slack.resize(n, 0);
	slackx.resize(n, 0);
	prev.resize(n, -1);

	if(org_cols == org_rows)
	{
		penalty = 0;
	}
	else
	{
		penalty = 0;
		for(int i = 0; i < org_rows; i++)
			for(int j = 0; j < org_cols; j++)
				penalty += input_matrix[i][j];
		penalty /= (org_rows*org_cols); 
	}

        for(int i = 0; i < n; i++)
        {
                for(int j = 0; j < n; j++)
                {
                        org_cost[i][j] =  (i < org_rows && j < org_cols) ? input_matrix[i][j] : penalty;

                        if(max_cost < cost[i][j])
                        {
                                max_cost = org_cost[i][j];
                        }
                }
        }

        if (mode == HUNGARIAN_MODE_MAXIMIZE_UTIL)
        {
              	// do nothing
		for(int i = 0; i < n; i++)
                {
                        for(int j = 0; j < n; j++)
                        {
                                cost[i][j] =  org_cost[i][j];
                        }
                }
        }
        else if (mode == HUNGARIAN_MODE_MINIMIZE_COST)
        {
                // This class deal with the maximum-weighted matching problem by default
		for(int i = 0; i < n; i++)
                {
                        for(int j = 0; j < n; j++)
                        {
                                cost[i][j] =  max_cost - org_cost[i][j];
                        }
                }
        }
        else
                cout << mode << " : unknown mode. Mode was set to HUNGARIAN_MODE_MINIMIZE_COST !\n";
}

void Assignment::init_labels()
{
    	for (int x = 0; x < n; x++)
        	for (int y = 0; y < n; y++)
            		lx[x] = max(lx[x], cost[x][y]);
}

void Assignment::update_labels()
{
    	int x, y, delta = INT_MAX;             //init delta as infinity
    	for (y = 0; y < n; y++)            //calculate delta using slack
    		if (!T[y])
    	        	delta = min(delta, slack[y]);
    	for (x = 0; x < n; x++)            //update X labels
        	if (S[x]) lx[x] -= delta;
    	for (y = 0; y < n; y++)            //update Y labels
        	if (T[y]) ly[y] += delta;
    	for (y = 0; y < n; y++)            //update slack array
        	if (!T[y])
            		slack[y] -= delta;
}

void Assignment::add_to_tree(int x, int prevx)
{ 
	//x - current vertex,prevx - vertex from X before x in the alternating path,
	//so we add edges (prevx, xy[x]), (xy[x], x)

    	S[x] = true;                    //add x to S
    	prev[x] = prevx;                //we need this when augmenting
    	for (int y = 0; y < n; y++)    //update slacks, because we add new vertex to S
	{
        	if (lx[x] + ly[y] - cost[x][y] < slack[y])
        	{
        	    	slack[y] = lx[x] + ly[y] - cost[x][y];
            		slackx[y] = x;
        	}
	}
}

void Assignment::augment()
{
    	if (max_match == n) return;        //check wether matching is already perfect
    	int x, y, root;                    //just counters and root vertex
    	int q[n], wr = 0, rd = 0;          //q - queue for bfs, wr,rd - write and read pos in queue

	S.assign((int)S.size(), false);
	T.assign((int)T.size(), false);
	prev.assign((int)prev.size(), -1);

    	for (x = 0; x < n; x++)            //finding root of the tree
        	if (xy[x] == -1)
        	{
            		q[wr++] = root = x;
            		prev[x] = -2;
            		S[x] = true;
            		break;
        	}

    	for (y = 0; y < n; y++)            //initializing slack array
    	{
        	slack[y] = lx[root] + ly[y] - cost[root][y];
        	slackx[y] = root;
    	}

	while (true)                                                        //main cycle
    	{
        	while (rd < wr)                                                 //building tree with bfs cycle
        	{
            		x = q[rd++];                                                //current vertex from X part
            		for (y = 0; y < n; y++)                                     //iterate through all edges in equality graph
                		if (cost[x][y] == lx[x] + ly[y] &&  !T[y])
                		{
                    			if (yx[y] == -1) break;                             //an exposed vertex in Y found, so augmenting path exists!
                    			T[y] = true;                                        //else just add y to T,
                    			q[wr++] = yx[y];                                    //add vertex yx[y], which is matched
                                                                        //with y, to the queue
                    			add_to_tree(yx[y], x);                              //add edges (x,y) and (y,yx[y]) to the tree
                		}
            		if (y < n) break;                                           //augmenting path found!
        	}
        	if (y < n) break;                                               //augmenting path found!

        	update_labels();                                                //augmenting path not found, so improve labeling
        	wr = rd = 0;                
        	for (y = 0; y < n; y++)        
        	//in this cycle we add edges that were added to the equality graph as a
        	//result of improving the labeling, we add edge (slackx[y], y) to the tree if
        	//and only if !T[y] &&  slack[y] == 0, also with this edge we add another one
        	//(y, yx[y]) or augment the matching, if y was exposed
            	if (!T[y] &&  slack[y] == 0)
            	{
                	if (yx[y] == -1)                                        //exposed vertex in Y found - augmenting path exists!
                	{
                    		x = slackx[y];
                    		break;
                	}
                	else
                	{
                    		T[y] = true;                                        //else just add y to T,
                    		if (!S[yx[y]])    
                    		{
                        		q[wr++] = yx[y];                                //add vertex yx[y], which is matched with
                        	                                                //y, to the queue
                        		add_to_tree(yx[y], slackx[y]);                  //and add edges (x,y) and (y,
                                                                        //yx[y]) to the tree
                    		}
                	}
           	}
        	if (y < n) break;                                               //augmenting path found!
    	}

    	if (y < n)                                                          //we found augmenting path!
    	{
        	max_match++;                                                    //increment matching
        	//in this cycle we inverse edges along augmenting path
        	for (int cx = x, cy = y, ty; cx != -2; cx = prev[cx], cy = ty)
        	{
        	    	ty = xy[cx];
            		yx[cy] = cx;
            		xy[cx] = cy;
        	}
       	 	augment();                                                      //recall function, go to step 1 of the algorithm
    	}
}//end of augment() function

Graph::Graph(int V)
{
	this->V = V;
	adj = new list<AdjListNode>[V];
}

void Graph::addEdge(int u, int v, float weight)
{
	AdjListNode node(v, weight);
	adj[u].push_back(node); // Add v to u's list
}

// A recursive function used by shortestPath. See below link for details
// http://www.geeksforgeeks.org/topological-sorting/
void Graph::topologicalSortUtil(int v, bool visited[], stack<int> &Stack)
{
	// Mark the current node as visited
	visited[v] = true;

	// Recur for all the vertices adjacent to this vertex
	list<AdjListNode>::iterator i;
	for (i = adj[v].begin(); i != adj[v].end(); ++i)
	{
		AdjListNode node = *i;
		if (!visited[node.getV()])
			topologicalSortUtil(node.getV(), visited, Stack);
	}

	// Push current vertex to stack which stores topological sort
	Stack.push(v);
}

// The function to find shortest paths from given vertex. It uses recursive
// topologicalSortUtil() to get topological sorting of given graph.
void Graph::shortestPath(int s, int d, float &minDist)
{
	stack<int> Stack;
	float dist[V];

	// Mark all the vertices as not visited
	bool *visited = new bool[V];
	for (int i = 0; i < V; i++)
		visited[i] = false;

	// Call the recursive helper function to store Topological Sort
	// starting from all vertices one by one
	for (int i = 0; i < V; i++)
		if (visited[i] == false)
			topologicalSortUtil(i, visited, Stack);

	// Initialize distances to all vertices as infinite and distance
	// to source as 0
	for (int i = 0; i < V; i++)
		dist[i] = FLT_MAX;
	dist[s] = 0;

	// Process vertices in topological order
	while (Stack.empty() == false)
	{
		// Get the next vertex from topological order
		int u = Stack.top();
		Stack.pop();

		// Update distances of all adjacent vertices
		list<AdjListNode>::iterator i;
		if (dist[u] != FLT_MAX)
		{
			for (i = adj[u].begin(); i != adj[u].end(); ++i)
				if (dist[i->getV()] > dist[u] + i->getWeight())
					dist[i->getV()] = dist[u] + i->getWeight();
		}
	}

	minDist = dist[d];

}

