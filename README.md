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
The results were evaluated using confusion matrices, training and validation loss graphs and ROC curve plots

![image](https://user-images.githubusercontent.com/62455043/213038263-1808e416-15c9-4b30-bb54-71e28e8e2b5b.png)

Confusion matrices for CNN with no data augmentation

![image](https://user-images.githubusercontent.com/62455043/213038601-ee3f2c86-206f-4d71-8d89-1b1c43a4f8fb.png)

Confusion matrices for CNN with data augmentation

![image](https://user-images.githubusercontent.com/62455043/213038628-32708e0c-856a-4553-ad50-1dbab89d96a0.png)

Training and validation graphs for CNN with no data augmentation

![image](https://user-images.githubusercontent.com/62455043/213038718-64393652-99d7-4104-b867-a686353b299b.png)

Training and validation graphs for CNN with data augmentation

![image](https://user-images.githubusercontent.com/62455043/213038749-36b6c565-9737-4667-b7f3-bf7e1fb4c6f4.png)

ROC Curve Plots for CNN Models in TensorFlow

![image](https://user-images.githubusercontent.com/62455043/213038899-3508d981-e54b-4b16-a077-f46df6ca5c5c.png)

ROC curve plot for DarkCovidNet

#### CNN table of results

| Model | Sensitivity | Specificity  | Accuracy |
| ------------- | ------------- |------------- | ------------- |
| Basic CNN-Adam  | 0.9748  | 1.0  | 0.9934  |
| DarkCovidNet-Adam  |  0.9804  |0.9969  | 0.9923  |
| AlexNet-SGD | 0.9958  |1.0  | 0.9989  |
| AlexNet-Adam  | 0.9497  |0.9940  | 0.9824  |
| Basic CNN-Adam with DA  |0.9414  | 0.997  | 0.9824  |
| AlexNet-SGD with DA	  | 0.9958  |0.9687  | 0.9758  |
| AlexNet-Adam with DA  | 0.9983  |0.9925  |  0.9901  |

With that, the models to be considered main were the AlexNet models with SGD and Adam optimizers. The DarkCovidNet did perform well overall, but since the implementation of the data augmentation was not possible, the results do not consider the same parameters and, therefore, are there just as benchmark.
Considering the contents of confusion matrices, train and validation loss graphs, ROC curve plots, and table of results the results show:
- Even with the additional data, the model performance is highly accurate, with a considerable enhance of 5% in sensitivity between AlexNet with and without Data Augmentation.
- The best overall performing data augmented model was the AlexNet utilizing the Adam optimizer, which showed considerably more accuracy (close to 2% more than the SGD counterpart), Specificity (also close to 2%) and marginally better performance in Sensitivity.
- The loss in training and validation with AlexNet using Adam was also considerably smaller, with little to no loss in the validation process, while SGD incurred into a considerable amount of noise. This further proves that SGD is more sensitive to noise in the data as shown in the training and validation graphs. 
- A good learning curve is generally visible to the eyes when the training loss plot reaches a point of stability, and the validation loss plot reaches a point of stability but has a small distance with the training loss. Training and validation graphs of AlexNet-SGD, AlexNet-Adam in, and AlexNet with DA show this trait.
- ROC Curves that are closer to the top-left of the plot dictate a good model and all models performed well on this as shown on the plot.
- DarkCovidNet-Adam, Basic CNN-Adam, AlexNet-SGD, and AlexNet-Adam with DA performed well with low misclassification rate from a visual perspective.
- Basic CNN-Adam, AlexNet-SGD,  and AlexNet-Adam with DA performed well with regards to sensitivity, specificity, and accuracy.
     Upon evaluating the results generated from the models, it was noted that the AlexNet-Adam with data augmentation model was deemed the recommended model. The model was notably a top performer based on the graphs and table above. With regards to using data augmentation, batch normalization, dropouts to avoid overfitting,  training and validation loss graph, confusion matrix and metrics, the AlexNet-Adam with DA data augmentation model is the chosen model.





