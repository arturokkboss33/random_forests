/* =============================================================*/
/* --- DECISION TREES - BINARY TREE NODE  SOURCE FILE        ---*/
/* FILENAME: NCMF_bst.h 
 *
 * DESCRIPTION: source file for the struct object of a binary
 * search tree, modified for its use in a decision tree 
 * learning algorithm. 
 *
 * VERSION: 1.0
 *
 * CREATED: 03/16/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//headers
#include "ncmf_bst.h"
//c++ libraries
#include <iostream>
#include <sstream>

//constructor
NCMF_BST::NCMF_BST(NCMF_node* rootPtr)
{
	root = rootPtr;
}

//get and set methods
NCMF_node* NCMF_BST::get_root() { return root;}

void NCMF_BST::set_root(NCMF_node* rootPtr)
{ root = rootPtr;}

//insert is not the same as in a common binary balance tree
//the logic to insert the node in a branch of the tree is handled in the decision tree learning algorithm
void NCMF_BST::insert_node(NCMF_node** rootPtr, std::string type, unsigned int idx, int depth, int classification, std::map<int, cv::Mat> left_centr, std::map<int, cv::Mat> right_centr, const cv::Mat& left_data, const cv::Mat& right_data, const cv::Mat& left_labels, const cv::Mat& right_labels)
{
	//check if the tree is empty
	if(*rootPtr == NULL)
	{
		*rootPtr = create_node(type, idx, depth, classification, left_centr, right_centr, left_data, right_data, left_labels, right_labels);
	}
	else
	{
		//normally logic to insert the node in the 'appropriate branch'
		//would be here, but as described this is handled by another program 
		std::cout << "\n Node could not be created" << std::endl;
	}

}

void NCMF_BST::insert_node(NCMF_node** rootPtr, NCMF_node* node)
{
	//check if the tree is empty
	if(*rootPtr == NULL)
	{
		*rootPtr = node;
	}
	else
	{
		//normally logic to insert the node in the 'appropriate branch'
		//would be here, but as described this is handled by another program 
		std::cout << "\n Node could not be created" << std::endl;
	}
}

//the next methods are used to print the decision tree
//the inOrder traversal and postOrder traversal are necessary
//to retrieve a unique strucuture of the tree
void NCMF_BST::inOrder(NCMF_node* ptr)
{
	if(ptr != NULL)
	{
		inOrder(ptr->f);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx <<  " ";
	
		inOrder(ptr->t);
	}
}

void NCMF_BST::postOrder(NCMF_node* ptr)
{
	if(ptr != NULL)
	{
		postOrder(ptr->f);
		postOrder(ptr->t);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx << " ";
	}
}

//create a node for the decision tree
//depending if it is a leaf or a split node, the node's fields are
//filled out differently
NCMF_node* NCMF_BST::create_node(std::string type, unsigned int idx, int depth, int classification, const std::map<int, cv::Mat> left_centr, std::map<int, cv::Mat> right_centr, const cv::Mat& left_data, const cv::Mat& right_data, const cv::Mat& left_labels, const cv::Mat& right_labels)
{
	NCMF_node* new_node = new NCMF_node();
	new_node->type = type;
	new_node->node_idx = idx;
	new_node->depth = depth;

	//if the node is a split, the attribute or feature that caused the
	//split is stored
	if(!(type.compare("split")))
	{
		
		new_node->output_id = -1;
		new_node->left_centroids = left_centr;
		new_node->right_centroids = right_centr;
		new_node->left_data = left_data.clone();
		new_node->right_data = right_data.clone();
		new_node->left_labels = left_labels.clone();
		new_node->right_labels = right_labels.clone();

		//std::cout << "Node split inserted" << std::endl;
	}
	
	//if the node is a leaf, an output is stored which contains the 
	//classification of particular case
	else
	{
		
		new_node->output_id = classification;
		//if it is a terminal node, we store all the remaining examples in the left_data/label  matrix
		new_node->left_centroids = left_centr;
		new_node->right_centroids = right_centr;
		new_node->left_data = left_data.clone();
		new_node->right_data = cv::Mat();
		new_node->left_labels = left_labels.clone();
		new_node->right_labels = cv::Mat();

		//std::cout << "Node terminal inserted" << std::endl;
	}
	new_node->f = new_node->t = NULL;

	return new_node;
}

