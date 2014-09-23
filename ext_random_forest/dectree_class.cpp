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

#include "dectree_class.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <map>
#include <list>

//can make a initializer list for a default constructor, but this
//gives the possibility to the client to change these default
//values

//Note: cv::Mat has to initialized otherwise is not recognized as a class type when used
Dectree_class::Dectree_class()
{
	dbst = NULL;
	classes = cv::Mat(0,1,CV_16SC1);
	dectree_idx = 0;
	max_depth = 0;
	min_depth = 1000; //just for research purposes
	split_nodes_idx = 0;
	terminal_nodes_idx = 0;
	depth_limit = 0;
}


int Dectree_class::get_dectree_idx() { return dectree_idx;}

void Dectree_class::set_dectree_idx(int idx)
{ dectree_idx = idx;}

dectree_node* Dectree_class::get_root(){ return dbst.get_root();}

void Dectree_class::inOrder_tree()
{
	dbst.inOrder(dbst.get_root());
}
void Dectree_class::postOrder_tree()
{
	dbst.postOrder(dbst.get_root());
}

int Dectree_class::get_noLeaves() { return terminal_nodes_idx;}

int Dectree_class::get_noNodes() { return (split_nodes_idx+terminal_nodes_idx);}

int Dectree_class::get_maxDepth() { return max_depth;}

void Dectree_class::train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, int vars_per_node)
{
	//initialize random generator
	rng = cv::RNG(time(NULL));
	//determine the number of classes based on the training data
	get_classes(labels);
	//std::cout << "Classes:\n" << classes << std::endl;
	//make a vector giving an id to each attribute
	set_attributes(training_data);
	//for debbugging
	/*
	for(std::vector<int>::iterator it = attributes.begin(); it != attributes.end(); ++it)
		std::cout << *it << " ";
	std::cout << std::endl;
	*/
	//compute the initial impurity just fot debugging
	//double hgoal = compute_entropy(labels);
	//std::cout << "Initial entropy: " << hgoal << std::endl;
	//grow a decision tree
	depth_limit = depth_thresh;
	min_samples = samples_thresh;
	if(attributes.size() < (unsigned)vars_per_node)
		active_vars = attributes.size();
	else
		active_vars = vars_per_node;
	int depth = 1;
	dbst.set_root(learn_dectree(cv::Mat(),labels, training_data, depth));
	find_depth(dbst.get_root());
	std::cout << "min depth: " << min_depth << std::endl;
	//for debugging
	//print the obtained tree
	/*
	std::cout<< "\ninOrder traversal: " << std::endl;
	dbst.inOrder(dbst.get_root());
	std::cout << std::endl;
	std::cout<< "\npostOrder traversal: " << std::endl;
	dbst.postOrder(dbst.get_root());	
	std::cout << std::endl;
	*/

	//only for testing
	/*
	int a = plurality(labels);
	std::cout << "Best class: " << a << std::endl;
	bool b = check_classif(labels);
	std::cout << "All same class?: " << b << std::endl;
	dectree_split* t = best_split(attributes, training_data, labels);
	std::cout << "Best attr: " << t->attr_name << " Pos: " << t->attr_idx << std::endl;
	std::cout << t->neg_attr_labels << std::endl;
	std::cout << t->neg_attr_data << std::endl;
	std::cout << t->pos_attr_labels << std::endl;
	std::cout << t->pos_attr_data << std::endl;
	*/
}

//check number of classes; remember labels is a column vector
void Dectree_class::get_classes(const cv::Mat& labels)
{
	char flag_difval = 0;
	int k = 0;
	
	classes.push_back(labels.at<int>(0));
	for(int e = 1; e < labels.rows; e++)
	{
		flag_difval = 0;
		k = 0;
		while(flag_difval == 0 && k < classes.rows)
		{
			if(labels.at<int>(e) == classes.at<int>(k))
				flag_difval = 1;
			k++;
		}	
		if(!flag_difval)
			classes.push_back(labels.at<int>(e));
	}

}

//set a vector of ints identifying each attribute
void Dectree_class::set_attributes(const cv::Mat& training_data)
{
	for(int attr = 0; attr < training_data.cols; attr++)
		attributes.push_back(attr);

}

double Dectree_class::compute_entropy(const cv::Mat& labels)
{
	double entropy = 0.;
	std::map<int, unsigned int> samples_per_class;

	//create a hash table with the classes names as keys, they save the number
	//of samples that have that particular label
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		samples_per_class.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );

	for(int ex = 0; ex < labels.rows; ex++)
		samples_per_class[labels.at<int>(ex)] += 1;

	//compute entropy based on the previous counting
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		double label_prob = (double)samples_per_class[classes.at<int>(no_classes)]/labels.rows;
		//std::cout << label_prob << std::endl;

		if( fabs(label_prob-0.) > FLT_EPSILON && fabs(label_prob-1.) > FLT_EPSILON )
			entropy += label_prob*log2(label_prob);
	}

	entropy *= -1.;

	return entropy;
}

//shanon entropy
double Dectree_class::compute_erf_entropy(const cv::Mat& labels, const cv::Mat& neg_labels, const cv::Mat& pos_labels)
{
	//class entropy or entropy before split
	double class_entropy = compute_entropy(labels);
	//mutual information (information gain)
	double imp_neg_labels = ((double)neg_labels.rows/labels.rows)*compute_entropy(neg_labels);
	double imp_pos_labels = ((double)pos_labels.rows/labels.rows)*compute_entropy(pos_labels);
	double imp_attr = imp_neg_labels + imp_pos_labels;
	double info_gain = class_entropy - imp_attr;
	//split entropy
	double split_entropy = 0;
	double neg_prob = ((double)neg_labels.rows/labels.rows); //because of the learning singular cases we are sure
								 //labels.rows is greater than 0
	double pos_prob = ((double)pos_labels.rows/labels.rows);
	if( fabs(neg_prob-0.) > FLT_EPSILON && fabs(neg_prob-1.) > FLT_EPSILON )
		split_entropy += neg_prob*log2(neg_prob);
	if( fabs(pos_prob-0.) > FLT_EPSILON && fabs(pos_prob-1.) > FLT_EPSILON )
		split_entropy += pos_prob*log2(pos_prob);
	split_entropy *= -1;
	//shannon entropy
	double shannon_entropy = (2*info_gain)/(class_entropy+split_entropy);

	return shannon_entropy;
}

//algorithm to learn a decision tree
//it returns the root of the decision tree (or one of its subtrees in the recursion) 
//inputs: parent examples, current examples, list of remaining attributes

dectree_node* Dectree_class::learn_dectree(const cv::Mat& p_samples_labels, const cv::Mat& samples_labels, const cv::Mat& samples_data, int depth)
{
	dectree_split* split = new dectree_split();	

	//auxiliary variables
	//double h_curr = 0.;
	Dectree_BST dectree; //decision tree structure
	dectree_node* dectree_root = dectree.get_root();

	//check the base cases to get out of the recursion
	//if there are no more examples		
	if(samples_labels.empty())
	{
		//std::cout << "Case 1" << std::endl;
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, plurality(p_samples_labels));
		return dectree_root;
	}
	//if all examples have the same classification
	else if(check_classif(samples_labels))
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, samples_labels.at<int>(0));
		return dectree_root;
	}
	//if there are no more attributes 
	/*else if(attr.empty())
	{
		//std::cout << "Case 3" << std::endl;
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, plurality(samples_labels));
		return dectree_root;

	}*/
	else if(depth >= depth_limit || (unsigned)samples_labels.rows < min_samples) //if this case is hit, there are attributes and samples to analyze
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, plurality(samples_labels));
		return dectree_root;
	}
	else
	{
		//find the attrribute with the highest information gain
		//h_curr = compute_entropy(samples_labels);
		//std::cout << "Current entropy: " << h_curr << std::endl;	
		split = erf_split(samples_data, samples_labels);
		std::cout << "Best attribute: " << split->attr_name << std::endl;
		std::cout << "Cut point: " << split->cut_point << std::endl;

		//create a tree with the best attribute as root
		dectree.insert_node(&dectree_root,"split", (++split_nodes_idx), depth, split->attr_name, split->cut_point, -1);	
		//std::cout << "Tree was splitted" << std::endl;

		//call the fcn recursively
		//for each of the classes of the attribute and the set
		//of examples created. Here, no attributes are erases
		//attr.erase(attr.begin()+split->attr_idx);
		(dectree_root)->f = learn_dectree(samples_labels,split->neg_attr_labels,split->neg_attr_data,(depth+1));	
		(dectree_root)->t = learn_dectree(samples_labels,split->pos_attr_labels,split->pos_attr_data,(depth+1));

		//std::cout << "Learning done" << std::endl;

	
		return dectree_root;	
	}
}

//Acoording to the basic cases of the learning algorithm
//To decide a classification output it's necessary to do a majority vote
//according to the remaining training examples
//Reminder: give samples matrix as a column vector
int Dectree_class::plurality(const cv::Mat& labels)
{

	std::map<int, unsigned int> samples_per_class;

	//create a hash table with the classes names as keys, they save the number
	//of samples that have that particular label
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		samples_per_class.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );

	for(int ex = 0; ex < labels.rows; ex++)
		samples_per_class[labels.at<int>(ex)] += 1;

	//iterate through the hash table to see which class has more votes
	//tied classes are kept in a vector
	std::map<int,unsigned int>::iterator it = samples_per_class.begin();
	int best_class = it->first;
	unsigned int max_votes = it->second;
	++it;
	cv::Mat tied_classes(0,1, CV_16SC1);
	tied_classes.push_back(best_class);
	
	for(; it != samples_per_class.end(); ++it)
	{
		if(it->second > max_votes)
		{
			best_class = it->first;
			max_votes = it->second;
			tied_classes.release();
			tied_classes.push_back(best_class);
		}
		else if(it->second == max_votes)
			tied_classes.push_back(it->first);
	}

	//if there are tied classes, pick one of them randomly
	//cv::RNG rng(time(NULL));
	if(tied_classes.rows > 1)
	{
		//std::cout << "Tie" << std::endl;
		int random_class = (int)(rng.uniform(0.,(double)tied_classes.rows));
		//Note: rng.uniform does not accept int values
		return tied_classes.at<int>(random_class);
	}
	else
		return best_class;	
}

//check if the remaining examples have the same classification
//if true, there is no necessity to continue the learning process
//through the respective branch
//Reminder: samples is a column vector
bool Dectree_class::check_classif(const cv::Mat& labels)
{

	bool flag_dif_class = false;
	int ex = 1;

	while(flag_dif_class == false && ex < labels.rows)
	{
		if(labels.at<int>(0) != labels.at<int>(ex))
			flag_dif_class = true;
		ex++;
	}

	return !flag_dif_class;

}

//return a 2-tuple indicating the best atr id and the its position of the list of all attributes (this position keeps changing as the list change size) 
//inputs: list of remaining attriutes, current examples, entropy of the ancestor node
dectree_split* Dectree_class::best_split(const cv::Mat& samples, const cv::Mat& labels)
{

	std::vector<int> attr = pick_variables();
	std::vector<int> best_attr_info(2,0);
	int true_attr_pos = 0;
	int attr_idx;
	double max_info_gain = -1.;
	double attr_info_gain = 0.;
	bool flag_compare_info_gain = false;
	cv::Mat final_neg_attr_data(0,1,CV_32FC1);
	cv::Mat final_pos_attr_data(0,1,CV_32FC1);
	cv::Mat final_neg_attr_labels(0,1,CV_16SC1);
	cv::Mat final_pos_attr_labels(0,1,CV_16SC1);
	cv::Mat neg_attr_labels(0,1,CV_16SC1);
	cv::Mat pos_attr_labels(0,1,CV_16SC1);

	double imp_bef_split = compute_entropy(labels);

	for(std::vector<int>::iterator it_attr = attr.begin(); it_attr != attr.end(); ++it_attr)
	{
		cv::Mat neg_attr_labels(0,1,CV_16SC1);
		cv::Mat pos_attr_labels(0,1,CV_16SC1);
		attr_idx = *it_attr;
		for(int ex = 0; ex < samples.rows; ex++)
		{
			if( fabs(samples.at<float>(ex,attr_idx)-0.) <= FLT_EPSILON)
				neg_attr_labels.push_back(labels.at<int>(ex));
			else
				pos_attr_labels.push_back(labels.at<int>(ex));
		}
		
		double imp_neg_attr_labels = ((double)neg_attr_labels.rows/samples.rows)*compute_entropy(neg_attr_labels);
		double imp_pos_attr_labels = ((double)pos_attr_labels.rows/samples.rows)*compute_entropy(pos_attr_labels);
		double imp_attr = imp_neg_attr_labels + imp_pos_attr_labels;
		attr_info_gain = imp_bef_split - imp_attr;
		//std::cout << "h1: " << imp_neg_attr_labels << std::endl;
		//std::cout << "h2: " << imp_pos_attr_labels << std::endl;
		//std::cout << "*** " << attr_info_gain;

		if(flag_compare_info_gain)
		{
			if(attr_info_gain > max_info_gain)
			{
				best_attr_info.at(0) = attr_idx;
				best_attr_info.at(1) = true_attr_pos;
				max_info_gain = attr_info_gain;
			}
		}
		else
		{
			flag_compare_info_gain = true;
			best_attr_info.at(0) = attr_idx;
			best_attr_info.at(1) = true_attr_pos;
			max_info_gain = attr_info_gain;
		}
		
		true_attr_pos++;
		neg_attr_labels.release();
		pos_attr_labels.release();

	}

	//std::cout << std::endl;
	//std::cout << "max: " << max_info_gain << std::endl;

	//with the found best attribute; fill the split structure
	//it's prefere to separate the samples again because otherwise a lot of memory reallocations take place
	for(int ex = 0; ex < samples.rows; ex++)
	{
		if( fabs(samples.at<float>(ex,best_attr_info.at(0))-0.) <= FLT_EPSILON)
		{
			final_neg_attr_data.push_back(samples.row(ex));
			final_neg_attr_labels.push_back(labels.at<int>(ex));
		}
		else
		{
			final_pos_attr_data.push_back(samples.row(ex));
			final_pos_attr_labels.push_back(labels.at<int>(ex));
		}
	}

	dectree_split* split = new dectree_split();
	split->attr_name = best_attr_info.at(0); 	//global idx of the attribute - to be use later for prediction
	split->attr_idx = best_attr_info.at(1);		//local idx of the attribute
	split->neg_attr_data = final_neg_attr_data;
	split->pos_attr_data = final_pos_attr_data;
	split->neg_attr_labels = final_neg_attr_labels;
	split->pos_attr_labels = final_pos_attr_labels;

	return split;

}

dectree_split* Dectree_class::erf_split(const cv::Mat& samples, const cv::Mat& labels)
{
	//select K attributes randomly
	std::vector<int> attr = pick_variables();
	//generate K random splits
	std::vector<double> attr_splits = gen_ran_splits(attr, samples);


	std::vector<int> best_attr_info(2,0);
	int true_attr_pos = 0;
	int attr_idx;
	double max_split_score = -1.;
	double best_cut_point = 0.;
	bool flag_compare_info_gain = false;
	cv::Mat final_neg_attr_data(0,1,CV_32FC1);
	cv::Mat final_pos_attr_data(0,1,CV_32FC1);
	cv::Mat final_neg_attr_labels(0,1,CV_16SC1);
	cv::Mat final_pos_attr_labels(0,1,CV_16SC1);
	cv::Mat neg_attr_labels(0,1,CV_16SC1);
	cv::Mat pos_attr_labels(0,1,CV_16SC1);

	//double imp_bef_split = compute_entropy(labels);

	//for(std::vector<int>::iterator it_attr = attr.begin(); it_attr != attr.end(); ++it_attr)
	for(std::vector<int>::size_type it_attr = 0; it_attr != attr.size(); it_attr++)
	{
		cv::Mat neg_attr_labels(0,1,CV_16SC1);
		cv::Mat pos_attr_labels(0,1,CV_16SC1);
		attr_idx = attr[it_attr];
		for(int ex = 0; ex < samples.rows; ex++)
		{
			//if( fabs(samples.at<float>(ex,attr_idx)-0.) <= FLT_EPSILON)
			if( samples.at<float>(ex,attr_idx) < attr_splits[it_attr] )
				neg_attr_labels.push_back(labels.at<int>(ex));
			else
				pos_attr_labels.push_back(labels.at<int>(ex));
		}
		
		double split_score = compute_erf_entropy(labels, neg_attr_labels, pos_attr_labels);
		std::cout << "*** entropy: " << split_score;

		//double imp_neg_attr_labels = ((double)neg_attr_labels.rows/samples.rows)*compute_entropy(neg_attr_labels);
		//double imp_pos_attr_labels = ((double)pos_attr_labels.rows/samples.rows)*compute_entropy(pos_attr_labels);
		//double imp_attr = imp_neg_attr_labels + imp_pos_attr_labels;
		//attr_info_gain = imp_bef_split - imp_attr;
		//std::cout << "h1: " << imp_neg_attr_labels << std::endl;
		//std::cout << "h2: " << imp_pos_attr_labels << std::endl;
		//std::cout << "*** " << attr_info_gain;

		if(flag_compare_info_gain)
		{
			if(split_score > max_split_score)
			{
				best_attr_info.at(0) = attr_idx;
				best_attr_info.at(1) = true_attr_pos;
				best_cut_point = attr_splits[it_attr];
				max_split_score = split_score;
			}
		}
		else
		{
			flag_compare_info_gain = true;
			best_attr_info.at(0) = attr_idx;
			best_attr_info.at(1) = true_attr_pos;
			best_cut_point = attr_splits[it_attr];
			max_split_score = split_score;
		}
		
		true_attr_pos++;
		neg_attr_labels.release();
		pos_attr_labels.release();

	}

	//std::cout << std::endl;
	//std::cout << "max: " << max_info_gain << std::endl;

	//with the found best attribute; fill the split structure
	//it's prefere to separate the samples again because otherwise a lot of memory reallocations take place
	for(int ex = 0; ex < samples.rows; ex++)
	{
		if( samples.at<float>(ex,best_attr_info.at(0)) < best_cut_point)
		{
			final_neg_attr_data.push_back(samples.row(ex));
			final_neg_attr_labels.push_back(labels.at<int>(ex));
		}
		else
		{
			final_pos_attr_data.push_back(samples.row(ex));
			final_pos_attr_labels.push_back(labels.at<int>(ex));
		}
	}

	dectree_split* split = new dectree_split();
	split->attr_name = best_attr_info.at(0); 	//global idx of the attribute - to be use later for prediction
	split->attr_idx = best_attr_info.at(1);		//local idx of the attribute
	split->cut_point = best_cut_point;
	split->neg_attr_data = final_neg_attr_data;
	split->pos_attr_data = final_pos_attr_data;
	split->neg_attr_labels = final_neg_attr_labels;
	split->pos_attr_labels = final_pos_attr_labels;

	return split;

}


std::vector<int> Dectree_class::pick_variables()
{
	std::map<unsigned int, short> chosen_attr;
	std::vector<int> picked_vars;
	int no_picked_vars = 0;

	//create a hash table
	//std::cout << attributes.size() << std::endl;
	for(unsigned int no_attr = 0; no_attr < attributes.size(); no_attr++)
		chosen_attr.insert( std::pair<int, short>(no_attr,0) );

	//cv::RNG rng(time(NULL));

	while(no_picked_vars < active_vars)
	{
		int r = (int)(rng.uniform(0., (double)attributes.size()));
		if(chosen_attr[r] == 0)
		{
			picked_vars.push_back(r);
			chosen_attr[r] += 1;
			no_picked_vars++;
		}
	}

	//for debugging
	std::cout << "Selected variables:" << std::endl;
	for ( std::vector<int>::iterator it=picked_vars.begin(); it != picked_vars.end(); ++it ) 
	{
		std::cout << ' ' << *it;
	}
	std::cout << std::endl;

	return picked_vars;
	
}

std::vector<double> Dectree_class::gen_ran_splits(std::vector<int> attr, const cv::Mat& samples)
{
	std::vector<double> ran_splits;
	for ( std::vector<int>::iterator it=attr.begin(); it != attr.end(); ++it ) 
	{
		double* max_val = new double();
		double* min_val = new double();
		cv::Mat attr_values = samples.col(*it).clone();
		cv::minMaxIdx(attr_values, min_val, max_val, NULL, NULL);
		double cut = rng.uniform(*min_val, *max_val);
		ran_splits.push_back(cut);	
		std::cout << "min: " << *min_val << " max: " << *max_val << " cut: " << cut << std::endl;	
	}

	return ran_splits;
	
}

int Dectree_class::predict(const cv::Mat& sample)
{
	dectree_node* tmp_node = dbst.get_root();
	int prediction = -1;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
			return tmp_node->output_id;
		else
		{
			if( sample.at<float>(tmp_node->attribute_id) < tmp_node->cut_point)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction;


}

cv::Mat Dectree_class::predict_with_idx(const cv::Mat& sample)
{
	dectree_node* tmp_node = dbst.get_root();
	cv::Mat prediction_mat = cv::Mat(0,1,CV_16SC1);
	int prediction = -1;
	int id = -1;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
		{
			prediction = tmp_node->output_id;
			id = tmp_node->node_idx;
			prediction_mat.push_back(prediction);
			prediction_mat.push_back(id);
			prediction_mat = prediction_mat.reshape(0,1);
			return prediction_mat;
		}
		else
		{
			if( sample.at<float>(tmp_node->attribute_id) < tmp_node->cut_point)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction_mat;


}

void Dectree_class::find_depth(dectree_node* ptr)
{
	if(ptr != NULL)
	{
		find_depth(ptr->f);

		if(!((ptr->type).compare("terminal")))
		{
			if(ptr->depth > max_depth)
				max_depth = ptr->depth;
			if(ptr->depth < min_depth)
				min_depth = ptr->depth;
		}
		
		find_depth(ptr->t);
	}
	
}

 //*************************
