# CV-Final-Project
Computer Vision Social Distancing Final Project

There are two main components to this project:

## User-based homography selection
This is intended to be run locally, since Google Colaboratory does not support XWindows. This notebook uses the OpenCV `imshow` with a few modifications to allow for simple homography selection. The user should select 4 points in the image which represent a square on the ground plane. The order of selection should be:
1. bottom left
2. bottom right
3. top right
4. top left

Double clicking on a point in the image will create a dot, indicating the selected location. Once four points have been selected, the user can double click near points to shift them. Additionally, a green grid will be displayed to help guide selection. This grid may be difficult to work with at first, as poor selections will make strange vanishing points. However, ideal selections make it very clear where the ground plane is. Small adjustments should yield reasonable ground planes.

## Detection pipeline
Two notebooks are provided, preloaded with information for a simple run of some VIRAT example data. `people_detector.ipynb` draws circles on an overhead view without any indication of violating social distance while `people_detector_distancing_visualization.ipynb` changes circle color based on violations. These notebooks are meant to be run on Google Colaboratory to leverage the GPUs provided. 

