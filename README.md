# Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays

# Important Links

[Link to data](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

[Slideshow Presentation](https://github.com/wajeeh853/Detecting-Pneumonia-in-Chest-X-Rays-using-Convolutional-Neural-Networks-/blob/main/Phase4PDF.pdf)

[Jupyter Notebook Full Modeling Process](https://github.com/wajeeh853/Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays/blob/main/Phase4_Final_Notebook-2.ipynb)

# Project Overview
XYZ Healthcare System is looking to hire a data scientist to build a machine learning model to classify their radiology department's chest x-ray images and screen them for pneumonia.

To help XYZ Healthcare System with this problem, I built a Convolutional Neural Network to classify the chest x-ray images as pneumonia or non-pneumonia. This way, XYZ Healthcare System can streamline their radiology department and improve their efficiency.

## Scope & Data Used
This project used the Mooney Pneumonia dataset, which can be found on kaggle via [this link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia"). This dataset included 5856 unique images of chest x-rays depicting normal or pneumonia x-rays.
# Preprocessing
In order to preprocess the images, I used Keras' ImageDataGenerator to rescale the images' pixels (to a scale of 0 to 1), and resized the images to 224 x 224 px. Even though the x-rays seem to be in grayscale, their actual sizes have all 3 layers of a colored photograph (rbg). Thus, I decided to leave them alone instead of converting them to grayscale. Below are two of the preprocessed images.

![alt text](https://github.com/wajeeh853/Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays/blob/main/Images/pre_model_imgs.png "Logo Title Text 1")
# Model Analysis
My final model was a Convolutional Neural Network with 83% accuracy. It has 6 blocks which include 10 convolutional layers, 5 max pooling layers, and 3 fully-connected layers.

## Metric Used

I used accuracy to score this model because both false negatives and false positives have a cost in this situation. For example, we want to reduce the rate of false negatives because we do not want anyone with pneumonia to go untreated. However, we would not want too many false positives either because we wouldn't want to prescribe steroids or antibiotics to patients that did not need it. In this use-case, even with the model's prediction, the x-ray will be looked over with human eyes before making a final diagnosis. Therefore, we want want the overall accuracy of the model to be high to make this process the most efficient for the technician or doctor reviewing the x-ray. The more accurate the model, the less time they need to spend in review.

## Feature Importances

Using the Lime package, I was able to depict the features (edges) in the image that the model found most important in making its prediction. The image on the left was correctly identified as a non-pneumonia. The features that led to this accurate prediction are outlined in yellow. The image on the right was incorrectly identified as non-pneumonia when it was actually pneumonia. The features that led it to this inaccurate prediction are outlined in yellow.

![alt text](https://github.com/wajeeh853/Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays/blob/main/Images/lime_final_model.png "Logo Title Text 1")

From the image on the right, we can see that the model picked up a piece of the diaphragm as an important feature in making its prediction. Because of this, the high-intensity pixels from the diaphragm may be adding noise to the model.

## Model Validation 

Looking at the final model's validaition and training curves, we can see that the model converges around 97% accuracy.

![alt text](https://github.com/wajeeh853/Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays/blob/main/Images/final_acc.png "Logo Title Text 1")

The final model had an accuracy score of 83% when predicting new data (test set). The confusion matrix from the test set is depicted below. The model is much better at correctly identifying the normal class, and still struggles with correctly identifying the pneumonia class. However, it is overall predicting most of each class correctly.

![alt text](https://github.com/wajeeh853/Convolutional-Neural-Network-to-Predict-pneumonia-in-chest-x-rays/blob/main/Images/final_cm.png "Logo Title Text 1")

## Model Conclusion and Recommendation

In conclusion, this model is not perfect. It could use some more tuning to increase its accuracy score. However, it will be very useful in improving the efficiency in XYZ Healthcare System's radiology department. Here are my recommendations:

Use this model as a tool for efficiency. Implementing this model in a radiology setting as way to assist x-ray technicians in detecting pneumonia would be the best way to use this technology. For example, once the chest x-ray is taken, it can automatically give its prediction to the tech. The tech and/or doctor would then need less time to review the model's prediction and use their trained eye for a final diagnosis. This would speed up the efficiency of the entire department so that the doctors' and techs' time can be mostly spent on other tasks.

The model will likely work better if the x-ray technician crops out the diaphragm before feeding the image to the model to remove noise.

I would also recommend the x-ray tech saves the image as 224 x 224 px before feeing the image into the model.

## Future Work

If I had time to explore further, I would do the following techniques to improve this model's performance:

1.Try to retrain the model using smaller images (ex. 64x64 px or 32x32 px) to see if it helps eliminate noise.

2.Look at more preprocessing methods to remove the diaphragm and thus remove noise.

3.Freeze and unfreeze specific layers of the transfer learning models to fine-tune them.

4.Do more research on other methods to tune cnns specifically with chest x-ray/pneumonia classification.
