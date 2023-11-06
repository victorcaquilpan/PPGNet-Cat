import pytorch_lightning as pl
import torch.nn as nn
from .reid_models import ReidModel, ReidTrunkModel,ReidLimbsModel


### Load the main model
class ReidMainModel(pl.LightningModule):
    def __init__(self, backbone_model,number_classes, embedding_size,arcface = False):
        super(ReidMainModel, self).__init__()   
        
        # Set variables
        self.backbone_model = backbone_model
        self.number_classes = number_classes
        self.embedding_size = embedding_size
        self.arcface = arcface
        
        # Define manual optimization
        self.automatic_optimization = False
    
        # Define the model
        self.full_image_model = ReidModel(backbone_model=self.backbone_model,number_classes = self.number_classes,embedding_size = self.embedding_size, arcface = self.arcface)
        self.trunk_model = ReidTrunkModel(embedding_size=self.embedding_size)
        self.limbs_model = ReidLimbsModel()

        # Create a function to initializate bottleneck
        def weights_init_kaiming(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        # Create a function to initialize classifier
        def weights_init_classifier(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, std=0.001)
                if m.bias:
                    nn.init.constant_(m.bias, 0.0)

        # Define Bottleneck
        self.bottleneck = nn.BatchNorm1d(self.embedding_size)
        self.bottleneck.bias.requires_grad_(False) 
        self.classifier = nn.Linear(self.embedding_size, self.number_classes, bias=False)
        
        # Inititlizate weights
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self,full,trunk, left_leg, right_leg, left_thig, right_thig, left_shank, right_shank,front_tail,rear_tail,label = None):
        logits, embedding  = self.full_image_model(full, label)
        embedding_trunk = self.trunk_model(trunk)
        embedding_paw = self.limbs_model(left_leg, right_leg, left_thig, right_thig, left_shank, right_shank, front_tail, rear_tail)

        # Sum features
        trunk_features = embedding + embedding_trunk
        limbs_features = embedding + embedding_paw

        # Add a final bottleneck and classifier to trunk features
        trunk_feat = self.bottleneck(trunk_features)
        logits_trunk = self.classifier(trunk_feat)

        # Add a final bottleneck and classifier to limbs features
        limbs_feat = self.bottleneck(limbs_features)
        logits_limbs = self.classifier(limbs_feat)
            
        # Return
        return logits, logits_trunk, logits_limbs, embedding, trunk_features, limbs_features
    
