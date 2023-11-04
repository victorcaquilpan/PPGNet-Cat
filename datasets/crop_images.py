# Import libraries
import numpy as np
import cv2
import math
import torch

# To rotate limbs
def rotate(points, angle):
    angle = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array([[c_x + np.cos(angle) * (px - c_x) - np.sin(angle) * (py - c_x),
                c_y + np.sin(angle) * (px - c_y) + np.cos(angle) * (py - c_y)] for px, py in points]).astype(int)

# Now, I include a function to plot limbs
def limbs_extraction(full_image,coordinates_x,coordinates_y,keypoint1, keypoint2, resized_limb_image,transform_limb, mirror = False):

    # Get min and max
    min_x = min([coordinates_x[i-1] for i in [keypoint1,keypoint2]])
    max_x = max([coordinates_x[i-1] for i in [keypoint1,keypoint2]])
    min_y = min([coordinates_y[i-1] for i in [keypoint1,keypoint2]])
    max_y = max([coordinates_y[i-1] for i in [keypoint1,keypoint2]])

    # Get height of rectangle
    height = math.dist((coordinates_x[keypoint1-1],coordinates_y[keypoint1-1]),(coordinates_x[keypoint2-1],coordinates_y[keypoint2-1]))
    # Get center
    center = int((max_x - min_x)/ 2 + min_x), int((max_y - min_y)/ 2 + min_y)
    # Get the width
    width = height/3
    # Calculate the angle between the two points
    angle = np.arctan2(coordinates_y[keypoint2-1] - coordinates_y[keypoint1-1], coordinates_x[keypoint2-1] - coordinates_x[keypoint1-1]) * 180.0 / np.pi
    # Create a rectangle with the desired width and length
    rectangle = np.array([[-height / 2, -width / 2],
                      [height / 2, -width / 2],
                      [height / 2, width / 2],
                      [-height / 2, width / 2]], dtype=np.int16)
    # Rotate the rectangle using the rotation matrix
    rotated_rectangle = rotate(rectangle,angle=angle)
    # Translate the rotated rectangle to the center point
    translated_rectangle = rotated_rectangle  + np.array(center)

    mask = np.zeros_like(full_image)

    # Fill the polygon defined by limb in the mask
    cv2.fillPoly(mask, [np.round(translated_rectangle).astype(int)], (255, 255, 255))

    # Find the contours of the mask
    try:
        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(contours[0])
        # Crop the original image using the bounding box

        #full_image = cv2.bitwise_and(full_image, full_image, mask=mask)
        mask[mask == 255] = 1
        full_image = mask * full_image 
        cropped_image = full_image[y:y+h, x:x+w]

        # Apply transformation
        cropped_image = transform_limb(image = np.array(cropped_image))['image']
        # Flip horizontally, if they are mirrowed images
        if mirror:
            cropped_image = torch.flip(cropped_image, [2])
            
        return cropped_image
    except:
        return torch.zeros((3,resized_limb_image[0], resized_limb_image[1]), dtype=torch.float)
    
def distance_between_point_and_line(point1, point2, point3):
    # Calculate the slope of the line passing through point1 and point2
    if point1[0] == point2[0]:
        # If the line is vertical, use a special case to avoid division by zero
        xD = point1[0]
        yD = point3[1]
    else:
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        # Calculate the coordinates of the point D where the perpendicular line intersects the line passing through A and B
        xD = (point3[0] + m * point3[1] - m * point1[0] + point1[1]) / (m * m + 1)
        yD = m * (xD - point1[0]) + point1[1]

    # Calculate the distance between C and D
    distance = math.sqrt((point3[0] - xD) ** 2 + (point3[1] - yD) ** 2)
    return distance

# Now, I include a function to crop the trunk
def trunk_extraction(full_image,coordinates_x,coordinates_y,resized_trunk_image,transform_trunk, mirror = False):

    # Get min and max between the tail and nose
    min_x = min([coordinates_x[i] for i in [2,13]])
    max_x = max([coordinates_x[i] for i in [2,13]])
    min_y = min([coordinates_y[i] for i in [2,13]])
    max_y = max([coordinates_y[i] for i in [2,13]])

    # Get height of rectangle
    height = math.dist((coordinates_x[2],coordinates_y[2]),(coordinates_x[13],coordinates_y[13]))
    # Get center
    center = int((max_x - min_x)/ 2 + min_x), int((max_y - min_y)/ 2 + min_y)


    # Check the points which are useful to define the boundaries of the body (ears and shoulders)
    points_boundaries  = [(coordinates_x[point],coordinates_y[point]) for point in [0,1,3,5] if (coordinates_x[point],coordinates_y[point]) != (0,0)]

    # Ask for the distance to each one of these points
    distances = []
    for points in points_boundaries:
        distances.append(distance_between_point_and_line((coordinates_x[2],coordinates_y[2]),(coordinates_x[13],coordinates_y[13]),points))

    # The width will be the maximum distance
    width = max(distances)
    
    # Calculate the angle between the two points
    angle = np.arctan2(coordinates_y[13] - coordinates_y[2], coordinates_x[13] - coordinates_x[2]) * 180.0 / np.pi
    # Create a rectangle with the desired width and length
    rectangle = np.array([[-height / 2, -width / 2],
                      [height / 2, -width / 2],
                      [height / 2, width / 2],
                      [-height / 2, width / 2]], dtype=np.int16)
    # Rotate the rectangle using the rotation matrix
    rotated_rectangle = rotate(rectangle,angle=angle)
    # Translate the rotated rectangle to the center point
    translated_rectangle = rotated_rectangle  + np.array(center)

    mask = np.zeros_like(full_image)

    # Fill the polygon defined by limb in the mask
    cv2.fillPoly(mask, [np.round(translated_rectangle).astype(int)], (255, 255, 255))

    # Find the contours of the mask
    try:
        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(contours[0])
        # Crop the original image using the bounding box

        #full_image = cv2.bitwise_and(full_image, full_image, mask=mask)
        mask[mask == 255] = 1
        full_image = mask * full_image 
        cropped_image = full_image[y:y+h, x:x+w]

        # Apply transformation
        cropped_image = transform_trunk(image = np.array(cropped_image))['image']
        # Flip horizontally, if they are mirrowed images
        if mirror:
            cropped_image = torch.flip(cropped_image, [2])
            
        return cropped_image
    except:
        return torch.zeros((3,resized_trunk_image[0], resized_trunk_image[1]), dtype=torch.float)