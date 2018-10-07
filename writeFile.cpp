#include <iostream>
#include <fstream>
using namespace std;

int main() {

	ofstream file ("example.txt");

	file << "Some text!!!" << endl;

	file.close();
}
