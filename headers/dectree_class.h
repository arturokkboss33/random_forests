/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS HEADER FILE       ---*/
/* FILENAME: dectree_class.cpp 
 *
 * DESCRIPTION: header file for the struct object which implements
 * a decision tree learning algorithm.
 *
 * VERSION: 1.0
 *
 * CREATED: 03/18/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */


#ifndef DECTREE_CLASS_H
#define DECTREE_CLASS_H

#include "dectree_bst.h"

#include <string>
#include <vector>

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

struct dectree_split 
{
	int attr_name;
	int attr_idx;
	cv::Mat neg_attr_data;		//samples with 'negative' classification
	cv::Mat pos_attr_data;		//samples with 'positive' classification
	cv::Mat neg_attr_labels;	//negative samples' labels
	cv::Mat pos_attr_labels;	//positive samples' labels
};

class Dectree_class
{
	public:
		//---constructor---
		Dectree_class();
		//get and set methods
		int get_dectree_idx();
		void set_dectree_idx(int idx);
		dectree_node* get_root();
		int get_noLeaves();
		int get_noNodes();
		
		//---auxiliary methods---
		void train(const cv::Mat& training_data, const cv::Mat& labels);
		int predict(const cv::Mat& sample);
		void inOrder_tree();
		void postOrder_tree();

	private:

		cv::Mat classes;
		std::vector<int> attributes;
		Dectree_BST dbst;
		int dectree_idx;
		unsigned int split_nodes_idx;
		unsigned int terminal_nodes_idx;

		void get_classes(const cv::Mat& labels);
		void set_attributes(const cv::Mat& samples); //create a list of attributes ids
		double compute_entropy(const cv::Mat& labels);
		dectree_node* learn_dectree(const cv::Mat& p_samples, const cv::Mat& samples, const cv::Mat& samples_data, std::vector<int> attr);//learning decision tree algorithm
		int plurality(const cv::Mat& samples);//majority count/vote
		bool check_classif(const cv::Mat& samples);//check if all the examples have the same classification
		dectree_split* best_split(std::vector<int> attr, const cv::Mat& samples, const cv::Mat& labels); //see which attribute has more information gain

};

#endif 
