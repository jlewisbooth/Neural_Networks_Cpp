#include <iostream> 
#include <algorithm>
#include <fstream>
#include <typeinfo>
#include <string>
#include <vector>
#include <sstream>
#include "CreateData.h"
#include <cstdlib>
#include <limits>
#include <cstddef>
#include <math.h>
using namespace std;

class Neuron {
	vector<double> weights;
	vector<double>deltaWeights;
	vector<double>nextWeights;
	double neuron_Val;
	double updateFunction_Val;
	double delta_Output;
	static int countNeuron;	
public:
	Neuron(double int_Val = 1) { neuron_Val = int_Val; updateFunction_Val = int_Val; }
	void setNextWeight(const double weight, const unsigned index) { nextWeights[index] = weight; }
	void updateWeights() { weights = nextWeights; }
	void setVal(const double &val) { neuron_Val = val; }
	void setUpdateFunction_Val(const double &funcVal) { updateFunction_Val = funcVal; }
	double getVal() { return neuron_Val; }
	double getFuncVal() { return updateFunction_Val; }
	double getWeight(const unsigned index) { return weights[index]; }
	double getDeltaWeight(const unsigned index) { return deltaWeights[index]; }
	double setWeight(const double Weight, const unsigned index) { weights[index] = Weight; }
	double setDeltaWeight(const double Weight, const unsigned index) { deltaWeights[index] = Weight; }
	void setDeltaOutput(const double &delOut) { delta_Output = delOut;}
	double getDeltaOutput() { return delta_Output; }
	void initialWeights(const unsigned &numNeurons);
	void printWeights();
};

int Neuron::countNeuron = 0;

void Neuron::printWeights()
{
	for(double weight : weights)
	{
		cout << weight << " ";
	}
	cout << endl;
};

void Neuron::initialWeights(const unsigned &numNeurons)
{
	for(unsigned i=0;i<numNeurons;++i)
	{
		double random = (0.2 * rand()) / RAND_MAX;
		weights.push_back(random);
		nextWeights.push_back(random);
		deltaWeights.push_back(0.0);
	}
};

typedef vector<Neuron> Layer;

class Net {
	vector<Layer> m_Layers;
	vector<string> Types;
	double learningRate;
	double momentum;
	double errorVal;
	double tolerence;
public:
	Net(dataSet *pntr);
	void addError(double &error) { errorVal += error; }
	void resetError() { errorVal = 0; }
	double getError() { return 0.5 * errorVal; }
	vector<unsigned> createTopology(dataSet *pntr);
	vector<string> getTypes() { return Types; }
	vector<double> setTarget(string &Type);
	void setWeights();
	void updateWeights();
	void printNeuronWeights(unsigned layer, unsigned neuron);
	void feedForward(vector<double> &inputVals, vector<double> &Target);
	vector<double> feedForward(vector<double> &inputVals);
	void printLayerVals(unsigned layerNum);
	void backProp(vector<double> &targetVals);
	void setParameters(double &learn, double &mo, double &tol) { learningRate = learn; momentum = mo; tolerence = tol;}
	double updatingFunction(double &Val) { return (1 / ( 1 + exp(-Val))); }
	double diffFunction(double &update_Val) { return (update_Val * ( 1 - update_Val)); } 
};

vector<unsigned> Net::createTopology(dataSet *pntr)
{
	vector<unsigned> topology;

	unsigned hiddenSize = 6;

	unsigned inputSize = pntr->dataPoints[0].dataVector.size();

	for( Data datapoint : pntr->dataPoints)
	{
		if (find(Types.begin(), Types.end(), datapoint.type) == Types.end())
		{
			Types.push_back(datapoint.type);
		}
	}

	unsigned outputSize = Types.size();

	topology = {inputSize,hiddenSize,outputSize};

	return topology;
}

void Net::updateWeights()
{	
	for( unsigned Layer = m_Layers.size()-1;Layer>0;--Layer)
	{
		if(Layer == m_Layers.size()-1)
		{
			for(unsigned neuron = 0;neuron<m_Layers[Layer].size();++neuron)
			{
				m_Layers[Layer][neuron].updateWeights();
			}
		}
		else
		{
			for(unsigned neuron = 0;neuron<m_Layers[Layer].size()-1;++neuron)
			{
				m_Layers[Layer][neuron].updateWeights();
			}
		}
	}
}

void Net::backProp(vector<double> &targetVals)
{
	unsigned Layer = m_Layers.size()-1;
	double deltaFunc;
	for(unsigned i=0;i<m_Layers[Layer].size();++i)
	{
		double outputVal = m_Layers[Layer][i].getFuncVal();
		double deltaOutput = (targetVals[i] - m_Layers[Layer][i].getFuncVal()) * diffFunction(outputVal);
		m_Layers[Layer][i].setDeltaOutput(deltaOutput);
		for(unsigned j=0;j<m_Layers[Layer-1].size();++j)
		{
			double deltaWeight = learningRate * deltaOutput * m_Layers[Layer-1][j].getFuncVal();
			double momentumWeight = momentum * m_Layers[Layer][i].getDeltaWeight(j);
			double newWeight = m_Layers[Layer][i].getWeight(j) + deltaWeight + momentumWeight;
			m_Layers[Layer][i].setNextWeight(newWeight,j);
			m_Layers[Layer][i].setDeltaWeight(deltaWeight,j);
		}
	}
	for(unsigned hiddenLayer=Layer-1;hiddenLayer>0;--hiddenLayer)
	{
		for(unsigned i=0;i<m_Layers[hiddenLayer].size()-1;++i)
		{
			double outputVal = m_Layers[hiddenLayer][i].getFuncVal();
			double deltaFunc = diffFunction(outputVal);
			double sumDeltaLayer = 0.0;
			for(unsigned j=0;j<m_Layers[hiddenLayer+1].size();++j)
			{
				sumDeltaLayer += m_Layers[hiddenLayer+1][j].getDeltaOutput() * m_Layers[hiddenLayer+1][j].getWeight(i);
			}
			double deltaOutputs = sumDeltaLayer * deltaFunc;
			m_Layers[hiddenLayer][i].setDeltaOutput(deltaOutputs);
			for(unsigned k=0;k<m_Layers[hiddenLayer-1].size();++k)
			{
				double deltaWeight = learningRate * deltaOutputs * m_Layers[hiddenLayer-1][k].getFuncVal();
				double momentumWeight = momentum * m_Layers[hiddenLayer][i].getDeltaWeight(k);
				double newWeight = m_Layers[hiddenLayer][i].getWeight(k) + deltaWeight + momentumWeight;
				m_Layers[hiddenLayer][i].setNextWeight(newWeight,k);
				m_Layers[hiddenLayer][i].setDeltaWeight(deltaWeight,k);	
			}
		}
	}
}

void Net::printLayerVals(unsigned layerNum)
{
	cout << "(";
	for(unsigned neuron=0;neuron<m_Layers[layerNum].size();++neuron) {
		if(neuron < m_Layers[layerNum].size() -1)
		{
			cout << m_Layers[layerNum][neuron].getFuncVal() << ",";
		}
		else
		{
			cout << m_Layers[layerNum][neuron].getFuncVal();
		}
	}
	cout << ") ";
};

void Net::feedForward(vector<double> &inputVals, vector<double> &Target)
{
	double error = 0;

	if(inputVals.size() != m_Layers[0].size() - 1)
	{
		cout << "Network not set up properly! Size = " << inputVals.size() <<  endl;
	}
	else {
	//      ============ Input Values ===========
		for(int i=0;i<m_Layers[0].size();++i)
		{
			if(i == m_Layers[0].size()-1)
			{
			// ======== Set bias node to zero =========
			       double bias = 1;	
				m_Layers[0][i].setUpdateFunction_Val(bias);
			}
			else
			{
				m_Layers[0][i].setUpdateFunction_Val(inputVals[i]);
			}
		}
		for(unsigned i=1;i<m_Layers.size();++i)
		{
			if(i < m_Layers.size()-1)
			{
			// ============ Hidden Layers ==========
				for(unsigned j=0;j<m_Layers[i].size()-1;++j)
				{
					double sum = 0;
					for(unsigned k=0;k<m_Layers[i-1].size();++k)
					{
						sum += m_Layers[i-1][k].getFuncVal() * m_Layers[i][j].getWeight(k);
					}
					m_Layers[i][j].setVal(sum);
					sum = updatingFunction(sum);
					m_Layers[i][j].setUpdateFunction_Val(sum);
				}
			}
			else
			{
			// ========= Final Layer =========
				for(unsigned j=0;j<m_Layers[i].size();++j)
				{
					double sum = 0;
					for(unsigned k=0;k<m_Layers[i-1].size();++k)
					{	
						sum += m_Layers[i-1][k].getFuncVal() * m_Layers[i][j].getWeight(k);
					}
					m_Layers[i][j].setVal(sum);
					sum = updatingFunction(sum);
					m_Layers[i][j].setUpdateFunction_Val(sum);
					error = pow((Target[j]-sum),2);
					addError(error);
				}
			}
		}
	}
};

vector<double> Net::feedForward(vector<double> &inputVals)
{
	vector<double> Output;

	if(inputVals.size() != m_Layers[0].size() - 1)
	{
		cout << "Network not set up properly! Size = " << inputVals.size() <<  endl;
	}
	else {
	//      ============ Input Values ===========
		for(int i=0;i<m_Layers[0].size();++i)
		{
			if(i == m_Layers[0].size()-1)
			{
			// ======== Set bias node to zero =========
			       double bias = 1;	
				m_Layers[0][i].setUpdateFunction_Val(bias);
			}
			else
			{
				m_Layers[0][i].setUpdateFunction_Val(inputVals[i]);
			}
		}
		for(unsigned i=1;i<m_Layers.size();++i)
		{
			if(i < m_Layers.size()-1)
			{
			// ============ Hidden Layers ==========
				for(unsigned j=0;j<m_Layers[i].size()-1;++j)
				{
					double sum = 0;
					for(unsigned k=0;k<m_Layers[i-1].size();++k)
					{
						sum += m_Layers[i-1][k].getFuncVal() * m_Layers[i][j].getWeight(k);
					}
					m_Layers[i][j].setVal(sum);
					sum = updatingFunction(sum);
					m_Layers[i][j].setUpdateFunction_Val(sum);
				}
			}
			else
			{
			// ========= Final Layer =========
				for(unsigned j=0;j<m_Layers[i].size();++j)
				{
					double sum = 0;
					for(unsigned k=0;k<m_Layers[i-1].size();++k)
					{	
						sum += m_Layers[i-1][k].getFuncVal() * m_Layers[i][j].getWeight(k);
					}
					m_Layers[i][j].setVal(sum);
					sum = updatingFunction(sum);
					Output.push_back(sum);
				}
			}
		}
	}
	return Output;
};

void Net::printNeuronWeights(unsigned layer, unsigned neuron)
{
	m_Layers[layer][neuron].printWeights();
};

void Net::setWeights()
{
	for(unsigned i=1;i<m_Layers.size();++i)
	{
		for(unsigned j=0;j<m_Layers[i].size();++j)
		{
			int size = m_Layers[i-1].size();
			m_Layers[i][j].initialWeights(size);
		}
	}
};

Net::Net(dataSet *pntr) {

	vector<unsigned> topology = createTopology(pntr);

	for(unsigned i=0;i<topology.size();++i)
	{		
		m_Layers.push_back(Layer());
		if(i < topology.size()-1)
		{	
			for(unsigned j=0;j<=topology[i];++j)
			{
				Neuron neuron;
				m_Layers.back().push_back(neuron);
			}
		}
		else
		{	
			for(unsigned j=0;j<topology[i];++j)
			{
				Neuron neuron;
				m_Layers.back().push_back(neuron);
			}
		}
	}
};

vector<double> Net::setTarget(string &Type)
{
	vector<double> target;

	for(unsigned i=0;i<Types.size();++i)
	{	
		if(Type == Types[i])
		{
			target.push_back(1.0);
		}
		else
		{
			target.push_back(0.0);
		}
	}
	return target;
}

void randomShuffle( unsigned* shuffle, unsigned &length)
{
	int j = 0,temp;

	for(unsigned i=0;i<length;i++)
	{
		shuffle[i] = j++;
	}

	for(unsigned i=0;i<length;i++)
	{
		int randIndex = rand() % length;

		temp = shuffle[i];
		shuffle[i] = shuffle[randIndex];
		shuffle[randIndex] = temp;
	}
}

void normalize(dataSet* pntr, unsigned &length)
{
	vector<double> min, max;
	for(double variable : pntr->dataPoints[0].dataVector)
	{
		min.push_back(variable);
		max.push_back(variable);
	}

	for(unsigned i=1;i<length;++i)
	{
		for(unsigned j=0;j<min.size();++j)
		{
			if(pntr->dataPoints[i].dataVector[j] > max[j])
			{
				max[j] = pntr->dataPoints[i].dataVector[j];
			}
			if(pntr->dataPoints[i].dataVector[j] < min[j])
			{
				min[j] = pntr->dataPoints[i].dataVector[j]; 
			}
		}	
	}
	
	for(unsigned i=0;i<length;++i)
	{
		for(unsigned j=0;j<min.size();++j)
		{
			pntr->dataPoints[i].dataVector[j] = (pntr->dataPoints[i].dataVector[j] - min[j]) / max[j];
		}
	}

}

void seperate(unsigned* index,unsigned& size) {
	
	unsigned inc = size / 12, training = 7 * inc, validation = 3*inc, test = size - training - validation;

	index[0] = training;
	index[1] = validation;
	index[2] = test;

}

double accuracy(vector<double> &output, vector<double> &target)
{
	double acc = 0;
	double fraction = 1 / (double)output.size();
	
	for(unsigned i=0;i<output.size();++i)
	{
		acc += abs(target[i] - output[i]) * fraction;
	}

	return 1 - acc;
}

int main() {

	ofstream dataFile("error.txt");

	dataSet* pntr = Create();

	unsigned count = 0;

	unsigned seperateIndex[3];

	seperate(seperateIndex,pntr->size);

	normalize(pntr,pntr->size);

	double learningRate = 0.5, momentum = 0.4, tolerence = 10, cost;
	
	unsigned shuffle[(pntr->size)-1];

	randomShuffle(shuffle,pntr->size);

	Net my_Net(pntr);

	my_Net.setParameters(learningRate,momentum,tolerence);

	my_Net.setWeights();

	vector<double> Target, Input;
	
	do {
		cout << "==== Pass " << count++ << " ====" << endl;
		
		my_Net.resetError();

		for(unsigned i=0; i<seperateIndex[0];++i)
		{
			string Type = pntr->dataPoints[shuffle[i]].type;

			Target = my_Net.setTarget(Type);

			Input = pntr->dataPoints[shuffle[i]].dataVector;

			my_Net.feedForward(Input,Target);

			my_Net.backProp(Target);

			my_Net.updateWeights();

			if(count == 9999)
			{

				cout << "Input Values = " << "(" << Input[0] << "," << Input[1] << "," << Input[2] << "," << Input[3] << "), ";
				cout << " Data point " << i << ": Target Values = " << "(" << Target[0] << "," << Target[1] << "," << Target[2] << "), ";

				cout << "Output Values = ";

				my_Net.printLayerVals(2);

				cout << endl;
			}
		}
		
		double validation_Accuracy = 0;
		
		for(unsigned i=seperateIndex[0];i<seperateIndex[0] + seperateIndex[1];++i)
		{

			string Type = pntr->dataPoints[shuffle[i]].type;

			Target = my_Net.setTarget(Type);

			Input = pntr->dataPoints[shuffle[i]].dataVector;

			vector<double> Output = my_Net.feedForward(Input);

			validation_Accuracy += accuracy(Output,Target);

		}

		cout << "Validation Accuracy : " << (validation_Accuracy*100.0) / (double)seperateIndex[1] << "%" << endl;

		cost = my_Net.getError();

		cout << my_Net.getError() / pntr->size  << endl;
		
		dataFile << count << "," << my_Net.getError() / pntr->size << endl;

	} while(count < 10000 || cost > 1.5);
	
	dataFile.close();

	double test_Accuracy = 0;

	unsigned Start = seperateIndex[0] + seperateIndex[1];

	for(unsigned i = Start;i< Start + seperateIndex[2];++i)
	{	
		string Type = pntr->dataPoints[shuffle[i]].type;

		Target = my_Net.setTarget(Type);

		Input = pntr->dataPoints[shuffle[i]].dataVector;

		vector<double> Output = my_Net.feedForward(Input);

		test_Accuracy += accuracy(Output,Target);
	}

	cout << "Test Accuracy : " << (test_Accuracy*100.0) / (double)seperateIndex[2] << "%" << endl;
	
	/*	
	vector<double> input = {1,2,3,4};

	cout << "==== Feed Forward ====" << endl;

	my_Net.feedForward(input);
	
	cout << "First Layer: "; 
		
	my_Net.printLayerVals(0);

	cout << "Weights for top neuron in Output Layer : ";
	my_Net.printNeuronWeights(2,0);

	cout << "Second Layer: "; 
		
	my_Net.printLayerVals(1);

	cout << "Third Layer: "; 
		
	my_Net.printLayerVals(2);

	vector<double> output = {0,1};

	cout << "==== Back Propagation ====" << endl;

	my_Net.backProp(output);
	
	cout << "New Weights for top neuron in Output Layer : ";
	my_Net.printNeuronWeights(2,0);*/
}
