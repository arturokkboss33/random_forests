/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS SOURCE FILE       ---*/
/* FILENAME: dectree_class.cpp 
 *
 * DESCRIPTION: source file for the struct object which implements
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

#include "erf_class.h"

#include <iostream>
#include <stdlib.h>


//can make a initializer list for a default constructor, but this
//gives the possibility to the client to change these default
//values

//Note: cv::Mat has to initialized otherwise is not recognized as a class type when used
ERF_class::ERF_class()
{
	max_trees = 1;
	tree_idx = 0;
}

void ERF_class::train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, int vars_per_node, int no_trees)
{

	//initialize random generator
	rng = cv::RNG(time(NULL));
	max_trees = no_trees;

	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = new Dectree_class(rng);
		dectree->set_dectree_idx(tree);
		dectree->train(training_data, labels, depth_thresh, samples_thresh, vars_per_node);
		rng = dectree->get_rng();
		forest.push_back(dectree);		
	}

}


int ERF_class::predict(const cv::Mat& sample)
{
	Dectree_class* dectree_tmp = forest.at(0);
	cv::Mat classes = dectree_tmp->get_classes();
	std::map<int, unsigned int> class_count;

	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		class_count.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );
	
	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = forest.at(tree);
		int prediction = dectree->predict(sample);
		class_count[prediction] += 1;
	}

	bool flag_compare = false;
	int best_class;
	unsigned int max_votes;
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		if(flag_compare)
		{
			if(class_count[classes.at<int>(no_classes)] > max_votes)
			{
				best_class = classes.at<int>(no_classes);			
				max_votes = class_count[best_class];
			}
		}	
		else
		{
			best_class = classes.at<int>(no_classes);			
			max_votes = class_count[best_class];
			flag_compare = true;
		}		
	}

	return best_class;
}

cv::Mat ERF_class::predict_with_idx(const cv::Mat& sample)
{
	Dectree_class* dectree_tmp = forest.at(0);
	cv::Mat classes = dectree_tmp->get_classes();
	std::map<int, unsigned int> class_count;

	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		class_count.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );
	
	cv::Mat used_leaves(0,2,CV_32SC1);
	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = forest.at(tree);
		cv::Mat prediction = dectree->predict_with_idx(sample);
		used_leaves.push_back(prediction);
		class_count[prediction.at<int>(0)] += 1;
		prediction.release();
	}

	bool flag_compare = false;
	int best_class;
	unsigned int max_votes;
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		if(flag_compare)
		{
			if(class_count[classes.at<int>(no_classes)] > max_votes)
			{
				best_class = classes.at<int>(no_classes);			
				max_votes = class_count[best_class];
			}
		}	
		else
		{
			best_class = classes.at<int>(no_classes);			
			max_votes = class_count[best_class];
			flag_compare = true;
		}		
	}

	cv::Mat final_prediction = (cv::Mat_<int>(1,2) << best_class, max_votes);
	used_leaves.push_back(final_prediction); 
	
	return used_leaves;
}




