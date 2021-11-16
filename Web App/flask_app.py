import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import random
import cv2

from app import app
from flask import flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from torchvision import datasets, models
from PIL import Image
from torch.autograd import Variable
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Interactive Mode
plt.ion()

# Load VGG16 pre-trained model for the dog detector
VGG16 = models.vgg16(pretrained=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Image transformation on the valid dataset
data_transforms = {
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

data_dir = "static/dogImages"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['valid']}

# Get class names and number of classes from valid dataset
class_names = image_datasets['valid'].classes
n_classes = len(class_names)

# Load pretrained Model Transfer
model_transfer = models.densenet161(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False
num_ftrs = model_transfer.classifier.in_features
model_transfer.classifier = nn.Linear(num_ftrs, n_classes)

# Load trained model 'model_transfer100.pt', to generate this model execute all cells in 'dog_app.ipynb' from the main repo
model_transfer.load_state_dict(torch.load('model_transfer100.pt'))

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Render the Upload page
@app.route('/')
def load_page():
	return render_template('upload.html')

# Route to upload image from user and show on the screen
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

# Predict Breed
@app.route('/predict/<filename>', methods=['POST', 'GET'])
def predict_breed(filename):

    # Check if gpu support is available
    use_cuda = torch.cuda.is_available()

    def predict_breed_transfer(img_path):

        # load the image and return the predicted breed
        img = Image.open(img_path) # Load the image from provided path

        # Preprocess of the input image:
        #       * resize the img
        #       * normalize it
        #       * convert to a PyTorch Tensor

        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # Resize
        preprocess = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

        # Convert to tensor
        img_tensor = preprocess(img).float()
        img_tensor.unsqueeze_(0) 
        img_tensor = Variable(img_tensor) 

        if use_cuda:
            img_tensor = Variable(img_tensor.cuda())
        model_transfer.eval()

        # Returns a Tensor of shape (batch, num class labels)
        output = model_transfer(img_tensor) 
        output = output.cpu()

        # The prediction will be the index of the class label with the largest value
        predict_index = output.data.numpy().argmax() 

        return predict_index, class_names[predict_index], image_datasets['valid'].classes[predict_index]

    # Using Open CV, detect if there is a human face on the image (True or False)
    def face_detector(img_path):
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    # Load image function to use in the dog_detector_function
    def load_image(img_path):
        image = Image.open(img_path).convert('RGB')
        
        # resize to (244, 244) because VGG16 accept this shape
        in_transform = transforms.Compose([
                            transforms.Resize(size=(244, 244)),
                            transforms.ToTensor()])

        image = in_transform(image)[:3,:,:].unsqueeze(0)
        return image

    # VGG16 model predict (returns the class from the pretrained dataset)
    def VGG16_predict(img_path):

        img = load_image(img_path)

        if use_cuda:
            img = img.cuda()
        ret = VGG16(img)

        return torch.max(ret,1)[1].item() # predicted class index

    # If the index from VGG16_predict is between 151 and 268, returns True (dog detected)
    def dog_detector(img_path):

        index = VGG16_predict(img_path)

        if index >= 151 and index <= 268:
            return 1 # True

        else:
            return 0 # False

    path = "static/uploads/"

    # Check if the image has a dog
    is_dog = dog_detector(path+filename)

    # Check if the image has a human
    is_human = face_detector(path+filename)

    # Get predicted breed
    predicted_index, predicted_name, pred_breed = predict_breed_transfer(path+filename)

    breed = pred_breed[4:].replace("_", " ")

    # Get random predicted breed image from the valid dataset 
    subdir = '/'.join(['static/dogImages/valid', str(pred_breed)])
    path2 = random.choice(os.listdir(subdir))
    final = '/'.join(['/dogImages/valid', str(pred_breed)])
    final = '/'.join([final, path2])

    # Render everything passing is_dog, is_human, breed and the path for the images filename and final
    return render_template('predict.html', filename=filename, is_dog=is_dog, is_human=is_human, breed=breed, final=final)

# Render final report
@app.route('/report', methods=['POST', 'GET'])
def report():
    return render_template('report.html')

if __name__ == "__main__":
    app.run()