# Multilabel Image Classification CNN
# Fashion MNIST Classifier

## About this project:
This project is dedicated to exploring the potential of semi-supervised techniques in the development of a robust multilabel image classification model. The primary objective is to accurately identify alphanumeric characters within noisy images of handwritten text. The project leverages the Combo MNIST dataset, which contains both labeled and unlabeled data, to facilitate this exploration. Various convolutional neural network architectures are assessed, with ResNet50 demonstrating superior performance. The ultimate aim is to equip the model to excel in real-world multilabel image classification scenarios.

## Business Value:
The business value of this project is significant as it enhances the accuracy and reliability of multilabel image classification models, particularly in scenarios requiring precise character recognition in noisy images. This technology has versatile applications across industries, including healthcare for more accurate interpretation of handwritten medical records, finance for processing handwritten forms, logistics for package tracking, and sentiment analysis from handwritten comments. It streamlines data entry, improves data accuracy, and reduces manual processing efforts, leading to increased efficiency and better decision-making.

## Producing Results:
**Preprocessing of Images:** Our preprocessing involved applying a Gaussian blur to remove noise, eroding and dilating to disconnect close characters, and using bounding boxes to extract individual characters. The images were reconstructed by scaling and aligning the characters to maintain a consistent format. We tested various data augmentation techniques, including horizontal and vertical flips, 10-degree rotations, and increasing sharpness. These transformations were randomly applied to a 25% subset of the training data. This process improved performance, contributing to our overall accuracy.

**Model Architecture:** We initially developed our convolutional neural network (CNN) based on research implementations. However, we plateaued at around 71% accuracy. After exploring other techniques, we switched to using preexisting network architectures like VGGNet, ResNet, and AlexNet. ResNet50, with 50 deep layers and skip connections, proved to be the best trade-off between training time and performance. We implemented a ResNet50 model, employing a sigmoid function in the final layer for multilabel classification. To optimize the model, we tuned hyperparameters and employed bagging techniques to reduce overfitting. Single-model bagging and multiple-model bagging were employed, both contributing to an overall improvement of test accuracy.

**Model Results:** We employed the teacher-student paradigm in semi-supervised learning to attain our highest test accuracy, approximately 95%. This remarkable accuracy score was reached through the utilization of multiple-model bagging with ResNet 50 networks. We applied bootstrapping to our concatenated dataset, which comprised augmented data, unlabeled data, and the original dataset.

## Reproducing Results:
To reproduce the results, please open the Jupyter notebook and run the entire notebook from top to bottom.

To run the notebook, download both the notebook itself and the training/test files from the Kaggle competition (https://www.kaggle.com/c/mais202fall2021/). Ensure that all the files are placed in the same directory and then run the notebook from that location. For additional details, please refer to the project report provided above.

## Files uploaded:
- **Model.ipynb:** Contains all project code, including data preprocessing, model training, and result evaluation.

- **Report.pdf:** Presents a comprehensive report of the project.

## Statement of Contribution:
This project was conducted as a collaborative effort within a group of three members, with all team members contributing equally at every stage of the project.

