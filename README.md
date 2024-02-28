# Multilabel Image Classification CNN
# Fashion MNIST Classifier

## About this project:
This project is dedicated to exploring the potential of semi-supervised techniques in the development of a robust multilabel image classification model. The primary objective is to accurately identify alphanumeric characters within noisy images of handwritten text. Leveraging the Combo MNIST dataset, which contains both labeled and unlabeled data, the project aims to facilitate this exploration. The focus is on employing various convolutional neural network architectures, with ResNet50 emerging as the model that demonstrates superior performance. The ultimate aim is to equip the model to excel in real-world multilabel image classification scenarios, ensuring accuracy and efficiency in identifying characters within diverse and noisy images.

## Producing Results:
### Preprocessing of Images:
In the image preprocessing phase, I implemented a series of techniques to enhance my model's feature extraction capabilities. I applied a Gaussian blur to eliminate noise, utilized erosion and dilation to disconnect close characters, and employed bounding boxes for individual character extraction. To maintain a consistent format, I reconstructed images through scaling and alignment of characters. Various data augmentation techniques, such as horizontal and vertical flips, 10-degree rotations, and increased sharpness, were randomly applied to a 25% subset of the training data. 

### Models Used:
I initially developed a convolutional neural network (CNN) based on research implementations. However, I plateaued at around 71% accuracy. After exploring other techniques, I switched to using preexisting network architectures like VGGNet, ResNet, and AlexNet. ResNet50, distinguished by 50 deep layers and skip connections, proved to be the best trade-off between training time and performance. I implemented a ResNet50 model, incorporating a sigmoid function in the final layer for multilabel classification. To optimize the model, I tuned hyperparameters and employed bagging techniques to reduce overfitting. Both single-model bagging and multiple-model bagging were employed, contributing to an overall improvement in test accuracy.

### Teacher-Student Paradigm:
In my pursuit of the highest test accuracy, I adopted the teacher-student paradigm within the realm of semi-supervised learning. This paradigm was instrumental in achieving my highest test accuracy, approximately 95%. Key strategies included the utilization of multiple-model bagging, leveraging ResNet50 networks, and the application of bootstrapping to my concatenated dataset, comprising augmented data, unlabeled data, and the original dataset.

## Results:
The culmination of my efforts resulted in a remarkable test accuracy of approximately 95%. The key strategies underpinning this success were the utilization of multiple-model bagging, leveraging ResNet50 networks, and the application of bootstrapping to my concatenated dataset. This dataset comprised augmented data, unlabeled data, and the original dataset. Through these sophisticated techniques and the strategic utilization of a diverse dataset, my model demonstrated excellence in real-world multilabel image classification scenarios, showcasing its ability to accurately identify alphanumeric characters within noisy images.

## Business Value:
The outcomes of this project have direct implications for various business applications, especially in image recognition and classification. The developed model, known for its high accuracy and robust noise tolerance, is applicable to scenarios requiring precise alphanumeric character identification within noisy images. Its utility spans across automated document processing, industrial quality control, and image-based data extraction tasks, minimizing the manual effort required in extracting information from handwritten or printed characters. In industrial settings, its robustness to noise adds value to quality control applications, contributing to enhanced production efficiency. The model's versatility in handling diverse image datasets creates opportunities in healthcare, finance, and legal sectors for efficient data extraction. The overall value proposition of the project lies in effectively addressing real-world challenges and providing reliable, accurate solutions, leading to significant time and resource savings.

## Reproducing Results:
To reproduce the results, please open the Jupyter notebook and run the entire notebook from top to bottom.

To run the notebook, download both the notebook itself and the training/test files from the Kaggle competition (https://www.kaggle.com/c/comp-551-fall-2021/data). Ensure that all the files are placed in the same directory and then run the notebook from that location. For additional details, please refer to the project report provided above.

## Files uploaded:
- **Model.ipynb:** Contains all project code, including data preprocessing, model training, and result evaluation.
- **Report.pdf:** Presents a comprehensive report of the project.
