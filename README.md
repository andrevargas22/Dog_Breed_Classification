# Dog Breed Classification with PyTorch

### Project Overview

Welcome to my Udacity Machine Learning Engineer Nanodegree Capstone Project! This is a PyTorch Convolutional Neural Network (CNN) project that, given an image of a dog, the algorithm identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

### Datasets

There are two datasets used in this project, a Dog Image Dataset and a Human Image Datasets. Both are available for download here:

[Dog Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location "/dogImages".  The "/dogImages" folder should contain 133 folders, each corresponding to a different dog breed.

[Human Dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). Unzip the folder and place it in the repo, at location "/lfw". 

### Table of Contents

* /Web App Folder - To finish this project, I made a small web application using Flask framework that asks for an input from the users and outputs the dog breed of that image. All files required for this Web App are inside this folder.
* /haarcascades Folder - OpenCV provides many pre-trained face detectors, stored as XML files on github. In this folder is one of these detectors used in the project to find human faces in images.
* dog_app.ipynb - This is the notebook where the entire project was made and tested. It needs access to both datasets in order to work properly.
* dog_app.pdf - The same notebook in pdf format.
* Proposal.pdf - This is the original document send to Udacity with the proposal of the project.
* Report.pdf - This is my final report, explaining the entire process from start to finish

### Original Project Files

If you want to do the original project from scratch, clone the repository and navigate to the downloaded folder.

	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
