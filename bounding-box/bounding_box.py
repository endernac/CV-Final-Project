import numpy as np

def extract_traits(bounding_box):
    """ Given bounding box coordinates, extracts relevant bounding box traits.
        Returns a dictionary that stores the height, width, and center coordinate of the given
        bounding box.
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

def count_undistanced(box_list, method=0):
    """ Uses a simple average to determine the distance between bounding boxes of identified 
        objects within an image, given a list of bounding box coordinates.
        Inputs:
            box_list - list of bounding box coordinates, where each entry in the list is
                a set of bounding box coordinates, given as an ndarray in the form 
                [y1, x1, y2, x2].
            method - method used to count the number of undistanced people.
                    0: denotes the use of depth-ratio method, assuming average height of 5.4 ft.
        Returns:
            pairwise_distance_array - array that contains the pairwise distances between each 
                bounding box's center point and the other bounding boxes' center points.
            num_pairs - number of pairs of people in image not socially distanced.
            pair_ids - bounding box indexes of the people not socially distanced.
    """
    if len(box_list) == 0:
        print("No detections in this image!")
        return None, None, None
    else:
        # ASSUMING AVERAGE HEIGHT
        if method == 0:
            # create list to store all data
            res = []
            for entry in box_list:
                # extract traits into array
                res.append(extract_traits(entry))
            res = np.array(res)
            # calculate depth similarity metric, P (ratio of areas)
            areas = (res[:,0] * res[:,1]).reshape(1,len(res))
            P_matrix = areas / areas.T
            P = np.min(np.array([np.triu(P_matrix), np.tril(P_matrix).T]), axis=0)
            # calculate ID: euclidean distance between centers scaled by avg height
            heights = res[:,0].reshape(1,len(res))
            pairwise_avg_height = (heights + heights.T) / 2
            pairwise_distance_array = []
            for center in res[:,2:]:
                pairwise_distance_array.append(np.linalg.norm(center-res[:,2:], axis=1))
            pairwise_distance_array = np.array(pairwise_distance_array)
            ID = pairwise_distance_array / pairwise_avg_height
            # calculate ratio value
            distance_ratio = P * ID
            # count undistanced individuals and return pair ids
            condition1 = (np.isclose(distance_ratio,(6/5.4)*np.ones(distance_ratio.shape))) | (distance_ratio <= 6/5.4*np.ones(distance_ratio.shape))
            condition2 = (distance_ratio != 0)
            distance_bool = (condition1 & condition2) + (condition1 & condition2).T
            num_pairs = sum(sum(condition1 & condition2))
            pair_ids = np.argwhere((condition1 & condition2)==True)
            return num_pairs, pair_ids, pairwise_distance_array