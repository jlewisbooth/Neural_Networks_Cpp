
#ifndef __DATA_H_INCLUDED__
#define __DATA_H_INCLUDED__

//==========* Dependencies *============

#include <iostream> 
#include <fstream>
#include <typeinfo>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

//==========* Code *=============
struct Data {
	vector<double> dataVector;
	string type;
};

Data GetData(string &text) 
{

	vector<double> x;
	string y;
	stringstream ss(text);
	int i = 0;
        Data info;	
	while(ss.good()) 
	{
		if(i < 4)
		{ 
			i = i + 1;
			string substr;
			getline(ss,substr,',');
			double num = stod(substr);		
			x.push_back(num);	
		}
		else 
		{	 
			string substr;
			getline(ss,substr,',');
			y = substr;			
		}
	}
	info.dataVector = x;
	info.type = y;
	return info;
}
#endif
