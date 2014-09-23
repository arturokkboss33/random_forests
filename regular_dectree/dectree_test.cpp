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
	/*
	std::cout << train_samples << std::endl;
	std::cout << train_labels << std::endl;
	std::cout << test_samples << std::endl;
	std::cout << test_labels << std::endl;
	std::cout << test_labels.type() << std::endl;
	*/
	//++++++++++++++++++++++++++++++++++//

	//+++ CREATE A DECISION TREE +++//
	Dectree_class* dectree;	
	dectree = new Dectree_class();
	dectree->set_dectree_idx(3); //give an id to the tree
	//std::cout << dectree->get_dectree_idx() << std::endl;
	std::cout << "TRAINING DECISION TREE" << std::endl;
	int max_depth = 30;
	int min_samples = 10;
	dectree->train(train_samples, train_labels, max_depth, min_samples);
	//++++++++++++++++++++++++++++++++++//	

	//+++ TESTING DECISION TREE +++//
	double good_classif = 0;
	for(int ex = 0; ex < train_samples.rows; ex++)
	{
		int prediction = dectree->predict(train_samples.row(ex));
		//std::cout << "+++ " << prediction << std::endl;
		if(prediction == train_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/train_samples.rows;

	std::cout << "Training accuracy: " << good_classif << std::endl;

	good_classif = 0;
	cv::Mat used_leaves(0,2,CV_16SC1);
	cv::Mat a;
	for(int ex = 0; ex < test_samples.rows; ex++)
	{
		int prediction = dectree->predict(test_samples.row(ex));
		used_leaves.push_back(dectree->predict_with_idx(test_samples.row(ex)));
		//std::cout << "+++ " << prediction << std::endl;
		if(prediction == test_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/test_samples.rows;

	std::cout << "Testing accuracy: " << good_classif << std::endl;
	std::cout << used_leaves.rows << " " << used_leaves.cols << std::endl;
	cv::Mat idxs = used_leaves.col(1).clone();
	idxs = idxs.reshape(0,1);
	std::cout << "Used leaves in testing:\n" << idxs << std::endl;
	//++++++++++++++++++++++++++++++++++//	

	//+++ PRINT TREE STRUCTURE +++//
	std::cout << "Max depth: " << dectree->get_maxDepth() << std::endl;
	std::cout << "No. Leaves: " << dectree->get_noLeaves() << std::endl;
	std::cout << "No. Nodes: " << dectree->get_noNodes() << std::endl;
	std::cout << "\nInOrder traversal: " << std::endl;
	dectree->inOrder_tree();
	std::cout << std::endl;
	std::cout << "\nPostOrder traversal: " << std::endl;
	dectree->postOrder_tree();
	std::cout << std::endl;
	

	
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
