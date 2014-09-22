/*
 * =====================================================================================
 *
 *       Filename:  dectree_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/17/2014 01:28:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

//predefined classes
#include "dectree_class.h"
//c++ libraries
#include <iostream>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

//defines
#define TRAIN_PATH "training_sets/"
#define TEST_PATH "test_sets/"

int main ( int argc, char *argv[] )
{
	//Dectree_class dectree;
	//Dectree_BST dbst;
	//double hgoal = 0.;
	std::vector<bool> results; //save list of results
	
	//+++ LOAD TRAINING AND TEST SET +++//
	std::cout << "LOADING TRAINING AND TEST SET..." << std::endl;
	//variables to parse the console input and search te trainig and test files
	std::string train_path = TRAIN_PATH;
	train_path += argv[1];	
	std::string test_path = TEST_PATH;
	test_path += argv[2];	
	//for debugging
	std::cout << "Train path: " << train_path.c_str() << std::endl;
	std::cout << "Test path: " << test_path.c_str() << std::endl;
	//saving the data in to opencv matrix
	cv::Mat train_samples, train_labels;
	cv::Mat test_samples, test_labels;
	cv::FileStorage fstore_train_data(train_path.c_str(), cv::FileStorage::READ);
	fstore_train_data["train_samples"] >> train_samples;
	fstore_train_data["train_labels"] >> train_labels;
	fstore_train_data.release();
	cv::FileStorage fstore_test_data(test_path.c_str(), cv::FileStorage::READ);
	fstore_test_data["test_samples"] >> test_samples;
	fstore_test_data["test_labels"] >> test_labels;
	fstore_train_data.release();
	//for debugging
	std::cout << train_samples << std::endl;
	std::cout << train_labels << std::endl;
	std::cout << test_samples << std::endl;
	std::cout << test_labels << std::endl;
	std::cout << test_labels.type() << std::endl;
	//++++++++++++++++++++++++++++++++++//

	//+++ CREATE A DECISION TREE +++//
	//cv::Mat test = (cv::Mat_<int>(9,1) << 6, 6, 6, 6, 6, 6, 6, 6, 5);
	Dectree_class* dectree;	
	dectree = new Dectree_class();
	dectree->set_dectree_idx(3);
	std::cout << dectree->get_dectree_idx() << std::endl;
	dectree->train(train_samples, train_labels);	
	std::cout << "done" << std::endl;

	

	
	/* 
	//Load training set
	dectree.load_trainset(train_file);
	dectree.print_trainset();
	//Compute entropy of overall classification
	dectree.set_hgoal();
	hgoal = dectree.get_hgoal();
	std::cout << "\nGoal entropy: " << hgoal << std::endl;
	//Build decision tree from the examples	
	dectree.set_dectree();
	dbst.set_root(dectree.get_dectree());
	//traverse the dectree inOrder and postOrder to define its structure
	std::cout<< "\ninOrder traversal: " << std::endl;
	dbst.inOrder(dectree.get_dectree());
	std::cout << std::endl;
	std::cout<< "\npostOrder traversal: " << std::endl;
	dbst.postOrder(dectree.get_dectree());
	
	std::cout << std::endl;

	//print results
	results = dectree.test_cases(test_file);
	std::cout << "\n \% error: " << dectree.get_per_error() << std::endl;
	std::cout << "\nTruth table for every test case" << std::endl;
	for(std::vector<bool>::iterator it = results.begin(); it != results.end(); ++it)
	{
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	*/

	return 0;
}				/* ----------  end of function main  ---------- */
