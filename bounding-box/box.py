import numpy as np

def extract_traits(bounding_box):
    """ Given bounding box coordinates, extracts relevant bounding box traits.
        Returns a dictionary that stores the height, width, and center coordinate of the
        given bounding box.
        Inputs:
            bounding_box - bounding box coordinates, given as an ndarray in the form 
            [y1, x1, y2, x2].
        Returns:
            height - height of bounding box.
            width - width of bounding box.
            centerx - x-coordinate of center of bounding box.
            centery - y-coordinate of center of bounding box.
    """
    # calculate height (in pixels) by taking difference between top and bottom
    height = np.abs(bounding_box[2]-bounding_box[0])
    # calculate width (in pixels) by taking difference between left and right
    width = np.abs(bounding_box[3]-bounding_box[1])
    # calculate center (in pixel coords) by taking averages of x and y coordinates
    centerx = np.mean([bounding_box[1],bounding_box[3]])
    centery = np.mean([bounding_box[0],bounding_box[2]])
    return height, width, centerx, centery

def get_distances_simpleaverage(box_list):
    """ Uses a simple average to determine the distance between bounding boxes of 
        identified objects within an image, given a list of bounding box coordinates.
        Inputs:
            box_list - list of bounding box coordinates, where each entry in the list is
                a set of bounding box coordinates, given as an ndarray in the form 
                [y1, x1, y2, x2].
        Returns:
            avg_distance - the average distance between all the bounding box center points.
            distance_array - array that contains the pairwise distances between each 
                bounding box's center point and the other bounding boxes' center points.
    """
    if len(box_list) == 0:
        print("No detections in this image!")
        return None, None
    else:
        # create list to store all data
        res = []
        for entry in box_list:
            # extract traits into array
            res.append(extract_traits(entry))
        res = np.array(res)
        # calculate pixel to feet ratio based on average height of 5.4 feet
        avg_height = np.mean(res[:,0])
        ft2px_ratio = 5.4 / avg_height
        # calculate distance between each pair of center points
        distance_array = []
        for center in res[:,2:]:
            distance_array.append(np.linalg.norm(center-res[:,2:], axis=1))
        distance_array = np.array(distance_array) * ft2px_ratio
        # average distance
        avg_distance = sum(sum(distance_array))/(np.square(len(distance_array))-len(distance_array))
        return avg_distance, distance_array

def count_undistanced(distance_array):
    """ Counts the number of pairs of people violating social distancing norms (not being
        more than 6 feet apart) given an array containing the pairwise distances between 
        identified bounding boxes.
        Inputs:
            distance_array - array that contains the pairwise distances between each 
                bounding box's center point and the other bounding boxes' center points.
        Returns:
            num_pairs - number of pairs of people in image not socially distanced.
            pair_ids - bounding box indexes of the people not socially distanced.
    """
    if distance_array is None:
        print("No distance array provided.")
        return None, None
    else:
        # remove duplicates in pairwise array (get upper triangular boolean)
        distance_bool_singles = np.triu(distance_array < 6)
        distance_bool_singles[np.eye(len(distance_array), dtype=bool)] = False
        # count the number of pairs of people less than 6 feet apart
        num_pairs = sum(sum(distance_bool_singles))
        # find the index of people not distancing
        pair_ids = np.argwhere(distance_bool_singles==True)
        return num_pairs, pair_ids