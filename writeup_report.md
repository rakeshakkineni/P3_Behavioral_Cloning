# **Behavioral Cloning Project** 

## Writeup 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizes the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and  drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is based on nVidia Autonomous Car model as it was proven. My model has 5 Convolution Layers , 4 dense layers and a dropout layer before the final output layer. For all the convolution layers,'RELU' activation was used.

Model summary is shown below.

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param 
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 51
=================================================================
Total params: 347,749
Trainable params: 347,749
Non-trainable params: 0
'''

#### 2. Attempts to reduce overfitting in the model
To avoid over fitting following strategies were followed. 
- Data was collected by driving the car equal number of laps in clockwise and anticlockwise direction
- The model contains dropout layer in order to reduce overfitting (model.py lines 89). 
- The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15-17). 
- The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
- Number of training epochs were limited to 3.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. All the three images recorded by the simulator were used for training. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I have started with [nVidia Autonomous Car model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). nVidia model was modified accept the cropped images of size (320x65x3) and the last layer of the model was modified to generate one output.

I have captured one lap data by running the simulator in clock wise direction , these images were fed to the model. I saw that the vehicle was able to drive itself and was able to reach till first turn without crossing the road boundaries. Near the first turn vehicle fell into the pond on the first attempt, neverthless it was a good start. 

After this i have have collected more data by driving vehicle 
  - In Clock-Wise and Anti-Clock Wise 3 laps each.
  - Near the first turn for 3 times. 
and I have started using Center , Left and Right images. After these changes vehicle was able to cross the first turn without any issue , but at the second turn it started going towards the mud path.

It was very difficult to find a way to make the vehicle cross the second turn without leaving the road. I knew there was no issue with the model and had to collect more data. I should have collected data for 10 times assuming that my driving was not proper, each time i was driving vehicle 5 laps each clockwise and anit clock wise direction and driving the vehicle near every turn for 5 times. I have increased the data size to around 100,000 images without any success. 
I read a suggestion in Udacity by Kevin_La_Ra , shown below, to use recovery data and less number of images. 

This suggestion has really helped me finishing the project. Taking his suggestion i have reduced the data size to 68,000 images and have included recovery images. Following drive cycles were followed to collect the data 
    - Clock Wise Laps : 2 Laps
    - Anti Clock Wise Laps :2 Laps
    - Only Turns: 5 times at each turn
    - Recovery Throughout the Lap: 2 Laps
Dropout layer was added just before the final Fully Connected layer. Addition of the Dropout layer has made the drive wobbly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


####2. Final Model Architecture

The final model architecture (model.py lines 71-90) is visualized below.

Here is a  of the architecture
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded two laps on track in anti clock wise direction. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to comeback to the center lane even if it has crossed the road limits. These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

I then recorded vehicle driving at each turn for 5 times.

After the collection process, I had 68,000 images. 20% of the images were used as validation images.

I have implemented a generator (model.py lines 34-62) to copy the images based on the information in driving_log.csv file. A correction factor of 0.2 was used for getting steering angle for Left and Right images.

All the images were normalized and cropped before training or validation. Model was trained for 3 epochs and adam optimizer was used.

The file (Command_Prompt_Output.txt) shows the command line output of the training process. Validation Loss was 0.0372 on the final epoch. 
