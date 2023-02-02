# Objective

Given an image of dog, algorithm will identify an estimate of the canine’s breed. If supplied an image of human, the code will identify the resembling of dog breed.

## The strategy laid out for solving the problem:

Step 0: Import Datasets
Step 1: Detect humans
Step 2: Detect dogs
Step 3: Create a CNN to Classify Dog breeds(from scratch)
Step 4: Use a CNN to Classify Dog breeds(using transfer learning)
Step 5: Write your Algorithm
Step 6: Test your Algorithm
Step 0: Import Datasets


train_files, valid_files, test_files - numpy arrays containing file paths to images
train_targets, valid_targets, test_targets - numpy arrays containing onehot-encoded classification labels
dog_names - list of string-valued dog breed names for translating labels

### Step 1: Detect Humans:

loading filenames in shuffled human dataset


I use OpenCV’s implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. We have downloaded one of these detectors and stored it in the haarcascades directory.


Percentage of human faces in first 100 human_files is 100%
Percentage of human faces in first 100 dog_files is 11%
Although the result is not perfect but its acceptable

### Step 2: Detect Dogs:

For dog detector, I have used the pre trained Resnet50 network. The weights used were the standard ones for the imagenet dataset.


I need to preprocess our images in to correct tensor size for using this model.

Keras CNNs require a 4D array (which we’ll also refer to as a 4D tensor) as input, with shape (nb_samples,rows,columns,channels)

where nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively.

In the path_to_tensor function we are processing a single image, so the output is (1,224,224,3), where 1 image, 224 pixels wide, 224 pixels high, and 3 colours red, green and blue. The image is loaded using the PIL library, and converted to the size 224x224. the img_to_array method separates the colors to (224x224x3) and finally we add a dimension at the front using the numpy expand_dims function to obtain our (1,224,224,3).

The paths_to_tensor function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape (nb_samples,224,224,3) where nb_samples is the number of samples, or number of images, in the supplied array of image paths.


Final step in preprocessing images is to use preprocess_input function to perform below step:

Convert RBG channels to BGR by reordering the channels
Normalizes the pixels based on standards for use with pret rained imagenet models
Now that I have a way to format our image for supplying to ResNet-50, I are now ready to use the model to extract the predictions. This is accomplished with the predict method, which returns an array whose i-th entry is the model's predicted probability that the image belongs to the i-th ImageNet category. This is implemented in the ResNet50_predict_labels function below.We then use numpy’s argmax function to isolate the class with the highest probability and use imagenet’s dictionary to identify the name of the class.


While looking at the imagenet’s dictionary, you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151–268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. If the prediction is in range (151 to 268) return True.


## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Now that I have functions for detecting humans and dogs in images, we need a way to predict breed from images. In this step, I will create a CNN that classifies dog breeds.

Network consists of 4 convolution layers along with 4 max pooling layers to reduce the dimensionality and increase the depth. The filters used were 16,32,64 respectively. Flatten is used to flatten the matrix and feed forward to two dense layers with 256, 512 nodes respectively. Final dense layer will have 133 nodes since there are 133 dog classes and used softmax function to predict probability for each clas

Dropout layers have been added to reduce the possibility of overfitting.


I have used default settings with Adam optimizer used as optimizer for loss function.

Train CNN model for 20 epochs and save the best model. Model is trained with 6680 samples and validated on 835 samples.


Load the model with best validation loss and predict on test dataset to evaluate model performance.



Target was to achieve a CNN with >1% accuracy. The network achieved 9.56% accuracy on test dataset with out any parameter fine tuning or data augmentation.

## Step 4: Use a CNN to Classify Dog Breeds
In this section I will use pretrained models available for use with keras using transfer learning.

Bottleneck features is the concept of taking a pre-trained model and chopping off the top classifying layer, and then inputing this chopped model as the first layer into our model.


The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. I only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.



Similar to above, I will train the model with 6680 samples and validates on 835 samples and save the model with best validation loss.


Test accuracy is increased to 39.47% by using pre trained VGG 16 model as feature extractor.


## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
I will use bottleneck features from pretrained Resnet50 model(provided by Udacity) to build our CNN to identify dog breed from images.


Model architecture is shown below. I have added two Dense layers with 512, 1024 nodes respectively with relu activation. Final layer will have 133 nodes with softmax activation to predict probabilities for each dog breed class. Dropout layer has been added to reduce overfitting.


Total trainable params will be 700k. We will compile the model with categorical_crossentropy loss and sgd optimizer.


I will train the model on 6680 samples and validate the model on 835 samples. I will save the model with best validation loss for future use.


Above trained model has test accuracy of 82.53% which is better than previous model which are trained using VGG19 or CNN(without tranfer learning).


Next step is to implementing the model in to a function that can be used in our web application.


When I input an image path, bottleneck features for our pretrained model are applied to the image, then it is processed through fully connected network for predicting the dog breed.

I apply np.argmax function to find the class with highest probability and use the label to get the dog name from dog_names dictionary.

## Step 6: Write your Algorithm
Here I will create algorithm to determine whether the image contains a human, dog, or neither. Then,

if a dog is detected in the image, return the predicted breed.
if a human is detected in the image, return the resembling dog breed.
if neither is detected in the image, provide output that indicates an error.

## Step 7: Test Your Algorithm
Below are the sample predictions by model:







### Reflection:

Training time for image recognition model built from scratch will be higher and requires more hyper parameter tuning. But, I can make use of pre trained models like VGG16, VGG19, Resnet50 by inputting bottleneck features to our first layer. Transfer learning can be used as better feature extractor.
If below techniques are applied to model testing accuracy can be improved further:

Increasing the training time(epochs)
Data Augmentation
Tuning hyperparameters (like learning rate, optimizers, loss_functions,nodes in dense layers)
Machine Learning
Image Classification
Image Processing
Data Science




