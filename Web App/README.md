# Dog Breed Classification Web App

### Overview

The files on this folder contain my Web App for Udacity's Machine Learning Engineer Capstone Project. It should work fine but a few things have to be modified:

* Change paths: I uploaded all these files to Python Anywhere to host this site for a while, so all the paths in the code are refering to Python Anywhere folders. For example, the absolute path data_dir = "/home/andrevargas22/mysite/static/dogImages" must be changed to the relative path of the local machine accordingly. All paths must be adjusted in order to make it work.

* Add a PyTorch Model: In the original project notebook, the CNN model was trained in 100 epochs and I saved it in the file "model_transfer100.pt". This model is loaded in the web app using torch.load, so this file is required. Open the notebook in the main folder to see instructions on how to create such a file.

* Add images from the dataset: The folder "static/dogImages/valid" here is empty, but should contain the folders from the valid dog dataset. This is used my the app to show the user a sample from the valid dataset of the predicted breed.
