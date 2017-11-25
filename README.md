# P3_Behavioral_Cloning
This repository contains keras model to drive a game car in autonomous mode. UDACITY car simulator and drive.py are needed for data collection and autonomous driving. [Nvidia Autonomous driving model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) was used as basis for this implementation.

## Tool Used:
  1. Anaconda 4.3.25
  2. Keras 2.0.9
  2. Python 3.6.2
  3. Opencv 3.1.0
  4. ffmpeg 3.1.3
 
 
## Input Files: 
- UDACITY Car simulator was used to collect the images for training and validation. Around 68,000 images were used for training the model. Around 97% Validation accuracy was achieved with in 3 epochs.
- Autonomous Mode of UDACITY Car simulator was used for testing the trained model.  

## Output Files:
Using video.py the output images , obtained during autonomous driving, were converted to video and uploaded to this repository.

## Project File:
model.py : Designed model for behavioral cloning project.

drive.py : This is from UDACITY.
