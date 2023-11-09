# Import libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2
from .crop_images import trunk_extraction, limbs_extraction

# Create a Reid Dataset
class ReidDataset(Dataset):
    def __init__(self,data_directory: str,
                 transform_type: bool = False, 
                 subset: str = 'train',
                 resized_full_image: tuple = (256,512),
                 resized_trunk_image: tuple = (64,128),
                 resized_limb_image: tuple = (64,64), 
                 mirrored_images: bool = False,
                 include_cat_keypoints = False,
                 min_images_per_entity = None):
        
        # Save the train state
        self.subset = subset
        self.resized_full_image = resized_full_image
        self.resized_trunk_image = resized_trunk_image
        self.resized_limb_image = resized_limb_image
        self.mirrored_images = mirrored_images
        self.include_cat_keypoints = include_cat_keypoints
        self.min_images_per_entity = min_images_per_entity

        # Check folder 
        self.data_directory = data_directory    
        # Define the data origin
        if self.subset == 'train' or self.subset == 'val':
            self.data = self.data_directory.cat_training_dir
            labels = pd.read_csv(self.data_directory.cat_anno_train_file)
            labels['mirror'] = False
            # Increase data for mirror images if necessary
            if self.mirrored_images:
                mirror_labels = labels.copy()
                mirror_labels['mirror'] = True

                # Check last class
                last_class = max(labels['entityid'])
                mirror_labels['entityid'] = mirror_labels['entityid'] + last_class + 1

                # Create the final dataset
                labels = pd.concat([labels, mirror_labels], axis=0)
                labels.reset_index(drop=True, inplace=True)

            # If we want to tackle imbalance problem
            if self.min_images_per_entity != None:
                class_counts = labels['entityid'].value_counts()

                target_samples = self.min_images_per_entity
                oversampled_dfs = []

                for class_label, count in class_counts.items():
                    class_df = labels[labels['entityid'] == class_label]
                    if count < target_samples:
                        oversampled_df = class_df.sample(target_samples - count, replace=True)
                        oversampled_dfs.append(oversampled_df)

                # Concatenate the original DataFrame with the oversampled DataFrames
                labels = pd.concat([labels] + oversampled_dfs)


            if self.subset == 'train':
                # Split in train and val
                val_labels = labels.groupby('entityid').first().reset_index()
                mask_labels = labels['filename'].isin(val_labels['filename'])
                self.labels = labels[~mask_labels]

                # Extract the keypoints
                if include_cat_keypoints:
                    self.keypoints = pd.read_csv(self.data_directory.keypoints_train)
                else:
                    # Create an empty keypoint dataframe
                    self.keypoints = pd.DataFrame(columns=['img'])
            else:
                self.labels = labels.groupby('entityid').first().reset_index()     
        elif self.subset == 'test':
                self.data = self.data_directory.cat_testing_dir
                self.labels = pd.read_csv(self.data_directory.cat_anno_test_file)
            
        # Check potential transformations
        self.transform_type = transform_type
        # Define the transformation 
        if self.transform_type:
            self.transform = albu.Compose([
                albu.SomeOf([
                    albu.GaussianBlur(sigma_limit=(0,1.5)),
                    albu.GaussNoise(mean = 0, var_limit=(0.0, 2.55**2), per_channel=True, p = 0.5),
                    albu.augmentations.CoarseDropout(max_height=0.2, max_width=0.2,max_holes=4),
                    albu.PiecewiseAffine(scale=(0.01, 0.03)),
                    albu.Perspective(scale=(0.01, 0.1)),
                    ],n=2),
                albu.Sequential([
                    albu.augmentations.Resize(self.resized_full_image[0],self.resized_full_image[1]),
                    albu.ToFloat(max_value=255.0),
                    albu.Rotate(10,border_mode=cv2.BORDER_CONSTANT),
                    albu.augmentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
                    ToTensorV2()
                    ])
                ])
            
            # Define transformations for trunk
            self.transform_trunk = albu.Compose([
                albu.SomeOf([
                    albu.GaussianBlur(sigma_limit=(0,1.5)),
                    albu.GaussNoise(mean = 0, var_limit=(0.0, 2.55**2), per_channel=True, p = 0.5),
                    albu.augmentations.CoarseDropout(max_height=0.2, max_width=0.2,max_holes=4),
                    albu.PiecewiseAffine(scale=(0.01, 0.03)),
                    albu.Perspective(scale=(0.01, 0.1)),
                    ],n=2),
                albu.Sequential([
                    albu.augmentations.Resize(self.resized_trunk_image[0],self.resized_trunk_image[1]),
                    albu.ToFloat(max_value=255.0),
                    albu.Rotate(10,border_mode=cv2.BORDER_CONSTANT),
                    albu.augmentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
                    ToTensorV2()
                    ])
                ])
            
            # Define trasnformations for limbs
            self.transform_limb = albu.Compose([
                albu.SomeOf([
                    albu.GaussianBlur(sigma_limit=(0,1.5)),
                    albu.GaussNoise(mean = 0, var_limit=(0.0, 2.55**2), per_channel=True, p = 0.5),
                    albu.augmentations.CoarseDropout(max_height=0.2, max_width=0.2,max_holes=4),
                    albu.PiecewiseAffine(scale=(0.01, 0.03)),
                    albu.Perspective(scale=(0.01, 0.1)),
                    ],n=2),
                albu.Sequential([
                    albu.augmentations.Resize(self.resized_limb_image[0],self.resized_limb_image[1]),
                    albu.ToFloat(max_value=255.0),
                    albu.Rotate(10,border_mode=cv2.BORDER_CONSTANT),
                    albu.augmentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
                    ToTensorV2()
                    ])
                ])
            
        # Consider a default transformation
        else:
            # Define trasnformations for full image
            self.transform = albu.Compose([
            albu.ToFloat(max_value=255.0),
            albu.augmentations.Resize(self.resized_full_image[0],self.resized_full_image[1]),
            albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
            ToTensorV2()])

            # Define trasnformations for trunk
            self.transform_trunk = albu.Compose([
                albu.augmentations.Resize(self.resized_trunk_image[0],self.resized_trunk_image[1]),
                albu.ToFloat(max_value=255.0),
                albu.Rotate(10,border_mode=cv2.BORDER_CONSTANT),
                albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
                ToTensorV2()
                ])
            
            # Define trasnformations for limbs
            self.transform_limb = albu.Compose([                
                    albu.augmentations.Resize(self.resized_limb_image[0],self.resized_limb_image[1]),
                    albu.ToFloat(max_value=255.0),
                    albu.Rotate(10,border_mode=cv2.BORDER_CONSTANT),
                    albu.augmentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
                    ToTensorV2()
                    ])

        # Define the transformation for test
        self.transform_test = albu.Compose([
            albu.ToFloat(max_value=255.0),
            albu.augmentations.Resize(self.resized_full_image[0],self.resized_full_image[1]),
            albu.transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), max_pixel_value=1.0),
            ToTensorV2()])
 
    def __getitem__(self,index):
        # Chose images. img_tuple contains image's path and class.
        entity = self.labels.iloc[index]

        # Get the class and the image contain for a first image

        img_tuple = entity['entityid'], entity['filename']

        # Read the img
        img = cv2.imread(self.data + img_tuple[1])
        try:
            # Try to transform
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        # Save the label
        label = img_tuple[0]
        # Implementing transformations
        if (self.subset == 'train'):
            img = self.transform(image = np.array(img))['image']
        
        elif (self.subset == 'test') or (self.subset == 'val'):
            img = self.transform_test(image = np.array(img))['image']

        # Check if we need to return the original or mirror entity:
        if (self.subset == 'train' or self.subset == 'val') and entity['mirror'] == True:
            img = torch.flip(img, [2])

        # Return trunk and limbs
        if self.subset == 'train':

            # Now return the full image (from pose set)
            if self.include_cat_keypoints and (img_tuple[1] in np.unique(self.keypoints['img'])):
                # Extract the full image
                full_pose_image = cv2.imread(self.data + img_tuple[1])

                keypoints_img = self.keypoints.loc[self.keypoints['img'] == img_tuple[1]]

                # Extract x and y
                coordinates_x = []
                coordinates_y = []
                val_point = []
                for kp in range(1,18):
                    point = keypoints_img.loc[keypoints_img['kp'] == kp]
                    if len(point) == 1:
                        coordinates_x.append(int(point['x'].iloc[0]))
                        coordinates_y.append(int(point['y'].iloc[0]))
                        val_point.append(1)
                    else:
                        coordinates_x.append(0)
                        coordinates_y.append(0)
                        val_point.append(0)

                # Try to fix channels
                try:
                    full_pose_image = cv2.cvtColor(full_pose_image, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass

                # Check availability of keypoints for trunk
                if all([val_point[point-1] != 0 for point in [3,14]]) and any([val_point[point-1] != 0 for point in [1,2]]) and any([val_point[point-1] != 0 for point in [4,6]]):
                    
                    # Define the trunk
                    trunk_img = trunk_extraction(full_pose_image,coordinates_x,coordinates_y,self.resized_trunk_image,self.transform_trunk, entity['mirror'])

                # In the case of not having enough points for trunk, enter a black image
                else:
                    trunk_img = torch.zeros((3, self.resized_trunk_image[0], self.resized_trunk_image[1]), dtype=torch.float)
                
                # Return limbs
                # Crop the left leg
                if  all([val_point[point] != 0 for point in [5,6]]):
                    left_leg = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,6,7,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    left_leg = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                
                # Crop the right leg
                if  all([val_point[point] != 0 for point in [3,4]]):
                    right_leg = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,4,5,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    right_leg = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the left thig
                if  all([val_point[point] != 0 for point in [10,11]]):
                    left_thig = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,11,12,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    left_thig = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the right thig
                if  all([val_point[point] != 0 for point in [7,8]]):
                    right_thig = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,8,9,self.resized_limb_image, self.transform_limb, entity['mirror'])
                else:
                    right_thig = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the left shank
                if  all([val_point[point] != 0 for point in [11,12]]):
                    left_shank = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,12,13,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    left_shank = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the left shank
                if  all([val_point[point] != 0 for point in [8,9]]):
                    right_shank = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,9,10,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    right_shank = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the front tail
                if  all([val_point[point] != 0 for point in [13,15]]):
                    front_tail = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,14,16,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    front_tail = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                # Crop the rear tail
                if  all([val_point[point] != 0 for point in [15,16]]):
                    rear_tail = limbs_extraction(full_pose_image,coordinates_x,coordinates_y,16,17,self.resized_limb_image,self.transform_limb, entity['mirror'])
                else:
                    rear_tail = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                

            # In the case of not having keypoints at all
            else:
                trunk_img = torch.zeros((3, self.resized_trunk_image[0], self.resized_trunk_image[1]), dtype=torch.float)
                left_leg = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                right_leg = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                left_thig = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                right_thig = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                left_shank = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                right_shank = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)

                # Generate empty tail images
                front_tail = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)
                rear_tail = torch.zeros((3, self.resized_limb_image[0], self.resized_limb_image[1]), dtype=torch.float)

            return img.squeeze(0), label, trunk_img, left_leg, right_leg, left_thig, right_thig, left_shank, right_shank, front_tail, rear_tail
        else:
            return img.squeeze(0), label, int(img_tuple[1][:-4])
        
    def __len__(self):
        return self.labels.shape[0]
