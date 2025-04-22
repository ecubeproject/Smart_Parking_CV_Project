# Smart_Parking_CV_Project
This is a computer vision based project on empty parking lot detection. It is combination of image classification, object detection and image segmentation.
The project demonstrated the use of two algorithms a CNN developed from scratch and A vision transformer based model. 
Project was decveloped using the high quality PKLot dataset that is considered a benchmark in computer vision
Identification of empty parking slot requires image classification, object detection and often image segmentation. However, with PKlot dataset, we do not need to do object detection and image segmentation as , although images are of entire parking ground, having many parking lots the XML annotations for each parking lot in the ground is available with every parking ground image
Another great advantage with PKLot dataset is that it contains images taken in different weather conditions, lightings and camera angles. So, the models made by training on this dataset are highly generalizable.
Here there are around 14000 parking ground images , each containing several parking lots (XML Annotated). So this is a massive data-set with 7,50,000 images.
DUe high quality and huge size of dataset it is feasible to make train, test and validation sets and obtain nearly 100% accuracy, precision, recall and F1-score




