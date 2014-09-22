#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"


int main ( int argc, char *argv[] )
{
//Variables for parsing the data file
	std::string filename = "SPECT.train";
	std::string line;
	std::stringstream parse;
	int ssize = 100; //establish a buffer size to store attribute values,
			 //which for binary classification string are no bigger than 1
	char c[ssize];
	char delimiter = ',';

	//Variables to store the values in the data file
	std::vector<int> tmpcase;
	std::vector< std::vector<int> > training_set;

	cv::Mat sample(0, 1, CV_32FC1);
	cv::Mat labels(0, 1 , CV_16SC1);
	cv::Mat train_set;

	std::ifstream dataset_file(filename.c_str(), std::ios::in);

	if(!dataset_file)
	{
		std::cerr << "Cannot load training set file" << std::endl;
	}
	else
	{
		while( (getline(dataset_file, line))!= NULL )
		{
			parse << line;

			while( parse.getline(c,ssize,delimiter) )
			{
				tmpcase.push_back( (*c-'0') );
				sample.push_back( (float)(*c-'0') );
			}

			parse.str(""); //safety measure to erase previous contents
			parse.clear(); //clear flags to be able to read from it again

			training_set.push_back(tmpcase);
			tmpcase.clear(); 

			train_set.push_back(sample.reshape(0,1));
			labels.push_back((int)(sample.at<float>(0)));
			sample = cv::Mat();
			
		}
	}

	std::cout << train_set << std::endl;
	cv::FileStorage fstore_traindata("spect_train.yml",cv::FileStorage::WRITE);
	cv::Mat train_samples(train_set.colRange(1,train_set.cols));
	fstore_traindata << "train_samples" << train_samples;
	fstore_traindata << "train_labels" << labels;
	fstore_traindata.release();
	std::cout << train_samples << std::endl;
	std::cout << labels << std::endl;

	std::vector<int> tmp;
	for(std::vector< std::vector<int> >::iterator it = training_set.begin(); it != training_set.end(); ++it)
	{
		tmp = *it;
		for(std::vector<int>::iterator it2 = tmp.begin(); it2 != tmp.end(); ++it2)
		{
			std::cout << *it2 << " ";
		}
		std::cout << std::endl;
		tmp.clear();
	}

}

