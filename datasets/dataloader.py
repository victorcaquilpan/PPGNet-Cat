# Import libraries
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ReidDataset

# Create LightninDataModule Class
class ReidDataModule(pl.LightningDataModule):
    def __init__(self,data_directory: str,
                 batch_size_train = None,
                 batch_size_test = None,
                 batch_size_val = None,
                 transform: bool = False,
                 num_workers: int = 2,
                 size_full_image = None,
                 size_trunk_image = None,
                 size_limb_image = None,
                 mirrored_images = None,
                 include_cat_keypoints = None,
                 min_images_per_entity = None):
        
        self.data_directory = data_directory
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers  
        self.transform = transform
        self.resized_full_image = size_full_image
        self.resized_trunk_image = size_trunk_image
        self.resized_limb_image = size_limb_image
        self.mirrored_images = mirrored_images
        self.include_cat_keypoints = include_cat_keypoints
        self.min_images_per_entity = min_images_per_entity

    def prepare_data(self):
        pass
    def setup(self):
        if self.batch_size_train:
            self.train_dataset = ReidDataset(data_directory= self.data_directory,
                                            transform_type=self.transform,
                                            subset = 'train',
                                            resized_full_image=self.resized_full_image,
                                            resized_trunk_image = self.resized_trunk_image,
                                            resized_limb_image= self.resized_limb_image,
                                            mirrored_images= self.mirrored_images,
                                            include_cat_keypoints = self.include_cat_keypoints,
                                            min_images_per_entity = self.min_images_per_entity)
            
            self.val_dataset = ReidDataset(data_directory=self.data_directory,
                                            transform_type=self.transform,
                                            subset = 'val',
                                            resized_full_image=self.resized_full_image,
                                            mirrored_images=self.mirrored_images)
            
        if self.batch_size_test:
            self.test_dataset = ReidDataset(data_directory=self.data_directory,
                                            transform_type=self.transform,
                                            subset = 'test',
                                            resized_full_image=self.resized_full_image)
            
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,shuffle=True,batch_size=self.batch_size_train,num_workers=self.num_workers)
        return train_dataloader
    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset,shuffle=False,batch_size=self.batch_size_val,num_workers=self.num_workers)
        return val_dataloader
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,shuffle=False,batch_size=self.batch_size_test,num_workers=self.num_workers)
        return test_dataloader