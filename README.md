# Applying multiple CNNs for Covid19 detection
Applying different CNN combinations to classify Covid19 from Kaggle


## Description

The dataset was collected from [Kaggle](https://www.kaggle.com/datasets/nasifajwad/normalpneumoniacovid?select=Curated+X-Ray+Dataset) where it contained Covid19, Normal and Pneumonia chest scans.
The purpose of this project was to use multiple CNN with different combinations and parameters. The models used were:
- Basic CNN with Adam optimiser
- DarkCovidNet with Adam optimiser
- AlexNet with SGD optimiser
- Basic CNN with Adam optimiser and data augmentation
- AlexNet with SGD optimiser and data augmentation
- AlexNet with Adam optimiser and data augmentation

The pneumonia images were removed and to only focus on classifying Normal and Covid19 chest scans.



## Methodology & Environment
KDD methodology was chosen to implement the project.

The description of the machine used, and the packages used is mentioned. The specifications of the machine used to perform the code has 11th Gen Intel i9-11950H, 64GB ram, 2TB SSD storage, and a NVIDIA GeForce RTX 3080 Laptop GPU. Docker 4.13.1 was used to install the latest Python, TensorFlow GPU and Jupyter image as of 10th of December 2022. The python packages used were matplotlib, tensorflow, keras, sklearn, PIL, os, numpy, seaborn, shutil, pathlib, pandas, torch version 1.13.0+cu116, and fastai version 2.7.10. 

## Pre-processing & Transformation

![image](https://user-images.githubusercontent.com/62455043/213033395-824a007d-e7ff-490c-92fd-cd2755126d94.png)

Sample images of the dataset were viewed and checked. There were a total of 4,551 images which were broken down into 1,281 Covid19 images and 3,270 normal images. These were split into 80% training and 20% test.

![image](https://user-images.githubusercontent.com/62455043/213033707-d8174c11-0ff8-41e5-82e0-40723630ea0c.png)

In order to help with imbalance, data augmentation was performed on the Covid19 images to help balance and potentially alleviate issues down the road of classification

## Implementation & Data Mining

DarkCovidNet was implemented using Pytorch and FastAI based on [DarkCovidNet Kaggle](https://www.kaggle.com/code/mekarthikd/dark-covid-net-ct-dataset/script) and [DarkCovidNet article](https://medium.com/visionwizard/darkcovidnet-automated-detection-of-covid-19-with-x-ray-images-c4bfc29eb06c).

AlexNet was implemented using TensorFlow based on [AlexNet article](https://medium.com/visionwizard/darkcovidnet-automated-detection-of-covid-19-with-x-ray-images-c4bfc29eb06c).

The models had some similar parameters:
- Images were resized to 224x224 pixels with the exception of DarkCovidNet working with 256.
- Batch size was set to 128
- Epochs were set to 25

The CNN models were created and then visualised using visualkeras library.

![image](https://user-images.githubusercontent.com/62455043/213037230-6314a476-abf8-4705-9478-a84b81615bd3.png)

Basic CNN model

![image](https://user-images.githubusercontent.com/62455043/213037274-a86c542c-512b-4d7d-833e-c0b1b27993a5.png)

AlexNet model

![image](https://user-images.githubusercontent.com/62455043/213038138-b7fb42d9-8ca0-4d61-8206-c8e7b689266c.png)

FastAI/Pytorch did not have a visualisation model that didn't require converting from FastAI to ONNX model to Tensorflow. 
A print of the learner summary was used instead


## Results & Evaluation
![image](https://user-images.githubusercontent.com/62455043/213038263-1808e416-15c9-4b30-bb54-71e28e8e2b5b.png)




