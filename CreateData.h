
#ifndef __CREATEDATA_H_INCLUDED__
#define __CREATEDATA_H_INCLUDED__

//==========* Dependencies *============

#include <iostream> 
#include <fstream>
#include <typeinfo>
#include <string>
#include <vector>
#include <sstream>
#include "getData.h"
using namespace std;

//==========* Code *===============
struct dataSet 
{
	vector<Data> dataPoints;
	unsigned size;
	
};

dataSet* Create() 
{
	dataSet *List = new dataSet;
	ifstream file;
	file.open("iris.data");
	string text;
	unsigned j = 0;
	while(getline(file,text)) 
	{
		if(text == "") {continue;}
		Data vectore;
		vectore = GetData(text);
		List->dataPoints.push_back(vectore);
		++j;
	}
	List->size = j;
	return List;
}

#endif
