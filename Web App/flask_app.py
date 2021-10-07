# Flask Libraries
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# Pytorch Libraries
import numpy as np
import time
import copy
from glob import glob
import torch
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch.autograd import Variable
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# OpenCV libraries
import cv2

plt.ion()   # interactive mode

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def load_page():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/loading_model/<filename>', methods=['POST'])
def loading_model(filename):
    return render_template ("loading.html", filename=filename)


@app.route('/predict/<filename>', methods=['POST', 'GET'])
def predict_breed(filename):

    # All images are resized to 224x224 and normalized
    # Only training images receive further augmentation
    data_transforms = {
        'train': transforms.Compose([
    #        transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    data_dir = "/home/andrevargas22/mysite/static/dogImages"

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['valid']}

    class_names = image_datasets['valid'].classes
    n_classes = len(class_names)

    #Check if gpu support is available
    use_cuda = torch.cuda.is_available()

    model_transfer = models.densenet161(pretrained=True)
    for param in model_transfer.parameters():
        param.requires_grad = False
    num_ftrs = model_transfer.classifier.in_features
    model_transfer.classifier = nn.Linear(num_ftrs, n_classes)

    model_transfer.load_state_dict(torch.load('/home/andrevargas22/mysite/model_transfer100.pt'))

    def predict_breed_transfer(img_path):

        # load the image and return the predicted breed
        img = Image.open(img_path) # Load the image from provided path

        # Now that we have an img, we need to preprocess it.
        # We need to:
        #       * resize the img.
        #       * normalize it, as noted in the PyTorch pretrained models doc,
        #         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        #       * convert it to a PyTorch Tensor.
        #
        # We can do all this preprocessing using a preprocess.

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
        img_tensor = preprocess(img).float()
        img_tensor.unsqueeze_(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.
        img_tensor = Variable(img_tensor) #The input to the network needs to be an autograd Variable
        if use_cuda:
            img_tensor = Variable(img_tensor.cuda())
        model_transfer.eval()
        output = model_transfer(img_tensor) # Returns a Tensor of shape (batch, num class labels)
        output = output.cpu()
        predict_index = output.data.numpy().argmax() # Our prediction will be the index of the class label with the largest value.

        return predict_index, class_names[predict_index], image_datasets['valid'].classes[predict_index]

    def face_detector(img_path):
        face_cascade = cv2.CascadeClassifier('/home/andrevargas22/mysite/haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    VGG16 = models.vgg16(pretrained=True)

    def load_image(img_path):
        image = Image.open(img_path).convert('RGB')
        # resize to (244, 244) because VGG16 accept this shape
        in_transform = transforms.Compose([
                            transforms.Resize(size=(244, 244)),
                            transforms.ToTensor()]) # normalizaiton parameters from pytorch doc.

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        return image

    def VGG16_predict(img_path):

        img = load_image(img_path)

        if use_cuda:
            img = img.cuda()
        ret = VGG16(img)

        return torch.max(ret,1)[1].item() # predicted class index

    def dog_detector(img_path):

        index = VGG16_predict(img_path)

        if index >= 151 and index <= 268:
            return 1 # True

        else:
            return 0 # False

    path = "/home/andrevargas22/mysite/static/uploads/"

    is_dog = dog_detector(path+filename)
    is_human = face_detector(path+filename)

    predicted_index, predicted_name, pred_breed = predict_breed_transfer(path+filename)

    breed = pred_breed[4:].replace("_", " ")

    subdir = '/'.join(['/home/andrevargas22/mysite/static/dogImages/valid', str(pred_breed)])
    path2 = random.choice(os.listdir(subdir))
    final = '/'.join(['/dogImages/valid', str(pred_breed)])
    final = '/'.join([final, path2])

    return render_template('predict.html', filename=filename, is_dog=is_dog, is_human=is_human, breed=breed, final=final)

@app.route('/report', methods=['POST', 'GET'])
def report():
    return render_template('report.html')

if __name__ == "__main__":
    app.run()