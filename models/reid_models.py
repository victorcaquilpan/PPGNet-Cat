# Import libraries
import torch
from torchvision import models
import torch.nn as nn
from arcface import ArcMarginProduct
import pytorch_lightning as pl

# Create DL model
class ReidModel(pl.LightningModule):
    def __init__(self, backbone_model,number_classes, embedding,arcface):
        super(ReidModel,self).__init__()
        self.number_classes = number_classes
        self.embedding = embedding
        self.arcface =  arcface

        # Define the model
        self.backbone_model = backbone_model
        if self.backbone_model == 'resnet101':
            self.backbone = models.resnet101(weights="DEFAULT")
        elif self.backbone_model == 'resnet152':
            self.backbone = models.resnet152(weights="DEFAULT")
        elif self.backbone_model == 'resnet50':
            self.backbone = models.resnet50(weights="DEFAULT")
            
        # Modify the backbone
        num_filters = self.backbone.fc.in_features
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1])) 
        # Use the pretrained model to classify and change the output of last layer
        self.adaptor = nn.Linear(num_filters, self.embedding)
        self.bn = nn.BatchNorm1d(self.embedding)
        self.embedding_fc = nn.utils.weight_norm(nn.Linear(self.embedding,self.embedding,bias=False), name='weight')
        self.final_fc = nn.utils.weight_norm(nn.Linear(self.embedding, self.number_classes, bias=False),name='weight')
        
        # If arcface is needed
        if self.arcface:
            self.arcface_layer = ArcMarginProduct(in_features = self.embedding ,out_features = self.number_classes)


    def forward(self, x, target = None):
        representations = self.backbone(x)
        out = self.adaptor(representations.flatten(1))
        out = self.bn(out)
        out = self.embedding_fc(out)
        embedding = torch.nn.functional.normalize(out, p=2, dim=1)  # normalized embedding
        if self.arcface and target != None:
            classify = self.arcface_layer(embedding, target)
        else:
            classify = self.final_fc(embedding)
        return classify, embedding

# Create trunk DL model
class ReidTrunkModel(pl.LightningModule):
    def __init__(self):
        super(ReidTrunkModel,self).__init__()

        # Define the model
        self.backbone = models.resnet34(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:7])

        # Define gap
        self.gap = nn.AdaptiveAvgPool2d((1, 10))
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap6 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap7 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap8 = nn.AdaptiveAvgPool2d((1, 1))
        self.num_part = 10

    def forward(self, x):
        x_proc = self.backbone(x)
        body_feat = self.gap(x_proc)
        part = {}
        #get eight part feature
        for i in range(self.num_part):
            part[i] = body_feat[:, :, :, i]                    
        body_feature = torch.cat((part[0], part[1], part[2], part[3], part[4], part[5], part[6], part[7], part[8], part[9]), dim=1) 
        # Flatten to (Batches, 2560)
        body_feature = body_feature.view(body_feature.shape[0], -1)  

        # Put zeros if trunk is a black image
        for idx, trunk_image in enumerate(x):
            if (trunk_image == torch.zeros((3, 64,128) , dtype=torch.float, device = 'cuda:0')).all():
                body_feature[idx] = torch.zeros((2560), dtype=torch.float, device = 'cuda:0')

        return body_feature


# Create limbs DL model
class ReidLimbsModel(pl.LightningModule):
    def __init__(self):
        super(ReidLimbsModel,self).__init__()

        # Define the backbones
        self.left_leg_model = models.resnet34(weights="DEFAULT")
        self.left_leg_model = nn.Sequential(*list( self.left_leg_model.children())[:7])

        self.right_leg_model = models.resnet34(weights="DEFAULT")
        self.right_leg_model = nn.Sequential(*list( self.right_leg_model.children())[:7])

        self.right_thig_model = models.resnet34(weights="DEFAULT")
        self.right_thig_model = nn.Sequential(*list( self.right_thig_model.children())[:7])

        self.right_shank_model = models.resnet34(weights="DEFAULT")
        self.right_shank_model = nn.Sequential(*list( self.right_shank_model.children())[:7])

        self.left_thig_model = models.resnet34(weights="DEFAULT")
        self.left_thig_model = nn.Sequential(*list( self.left_thig_model.children())[:7])

        self.left_shank_model = models.resnet34(weights="DEFAULT")
        self.left_shank_model = nn.Sequential(*list( self.left_shank_model.children())[:7])

        # Add the model for the front tail
        self.front_tail_model = models.resnet34(weights="DEFAULT")
        self.front_tail_model = nn.Sequential(*list(self.front_tail_model.children())[:7])

        # Add the model for the rear tail
        self.rear_tail_model = models.resnet34(weights="DEFAULT")
        self.rear_tail_model = nn.Sequential(*list(self.rear_tail_model.children())[:7])
                                  
        # Layer 4 of resnet34
        self.base1 = models.resnet34(weights="DEFAULT")
        self.base1 = nn.Sequential(*list( self.base1.children())[7:8])
        self.base2 = models.resnet34(weights="DEFAULT")
        self.base2 = nn.Sequential(*list( self.base2.children())[7:8])
        self.base3 = models.resnet34(weights="DEFAULT")
        self.base3 = nn.Sequential(*list( self.base3.children())[7:8])
        self.base4 = models.resnet34(weights="DEFAULT")
        self.base4 = nn.Sequential(*list( self.base4.children())[7:8])

        # Inclusion of adaptive layers
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap6 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, l_leg,r_leg,l_thig,r_thig,l_shank,r_shank,f_tail,r_tail):

        part_feat = {}
        part_feat[1] = self.left_leg_model(l_leg)
        part_feat[2] = self.right_leg_model(r_leg)
        part_feat[3] = self.right_thig_model(r_thig)
        part_feat[4] = self.right_shank_model(r_shank)
        part_feat[5] = self.left_thig_model(l_thig)
        part_feat[6] = self.left_shank_model(l_shank)
        part_feat[7] = self.front_tail_model(f_tail)
        part_feat[8] = self.rear_tail_model(r_tail)

        # Create a clone of the outcomes to replace by zeros when correspond            
        proc_part_feat = {}
        for k,v in part_feat.items():
            proc_part_feat[k] = v.clone()

        for idx, img_thig in enumerate(r_thig):
            if (img_thig == torch.zeros((3, 64,64) , dtype=torch.float, device = 'cuda:0')).all():
                proc_part_feat[3][idx] = torch.zeros((256,4,4), dtype=torch.float, device = 'cuda:0')

        for idx, img_shank in enumerate(r_shank):
            if (img_shank == torch.zeros((3, 64,64) , dtype=torch.float, device = 'cuda:0')).all():
                proc_part_feat[4][idx] = torch.zeros((256,4,4), dtype=torch.float, device = 'cuda:0')

        for idx, img_thig in enumerate(l_thig):
            if (img_thig == torch.zeros((3, 64,64) , dtype=torch.float, device = 'cuda:0')).all():
                proc_part_feat[5][idx] = torch.zeros((256,4,4), dtype=torch.float, device = 'cuda:0')

        for idx, img_shank in enumerate(l_shank):
            if (img_shank == torch.zeros((3, 64,64) , dtype=torch.float, device = 'cuda:0')).all():
                proc_part_feat[6][idx] = torch.zeros((256,4,4), dtype=torch.float, device = 'cuda:0')

        # Fusion of parts. Thigs and shanks
        behind_top = torch.add(proc_part_feat[3], proc_part_feat[5])
        behind_down = torch.add(proc_part_feat[4], proc_part_feat[6])

        new_parts_x = [proc_part_feat[1], proc_part_feat[2], behind_top, behind_down,proc_part_feat[7],proc_part_feat[8]]
        new_part_feat = {}

        # Implement layer4 of Resnet34
        for i, x in enumerate(new_parts_x):
            if i == 0:
                x = self.base1(x)
                x = self.gap1(x)
                new_part_feat[i+1] = x
            if i == 1:
                x = self.base2(x)
                x = self.gap2(x)
                new_part_feat[i+1] = x
            if i == 2:
                x = self.base3(x)
                x = self.gap3(x)
                new_part_feat[i+1] = x
            if i == 3:
                x = self.base4(x)
                x = self.gap4(x)
                new_part_feat[i+1] = x
            #  For the tail, we are not including the base to keep them in a size of 256
            if i == 4:
                x = self.gap5(x)
                new_part_feat[i+1] = x
            if i == 5:
                x = self.gap6(x)
                new_part_feat[i+1] = x

        
        # Check again if some of the previous layers are zero. 

        # Put zeros if it is a black image    
        for idx, part_feat1 in enumerate(proc_part_feat[1]):
            if (part_feat1 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[1][idx] = torch.zeros((512,1,1), dtype=torch.float, device = 'cuda:0')

        # Put zeros if it is a black image    
        for idx, part_feat2 in enumerate(proc_part_feat[2]):
            if (part_feat2 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[2][idx] = torch.zeros((512,1,1), dtype=torch.float, device = 'cuda:0')

        # Put zeros if it is a black image    
        for idx, part_feat3 in enumerate(behind_top):
            if (part_feat3 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[3][idx] = torch.zeros((512,1,1), dtype=torch.float, device = 'cuda:0')

        # Put zeros if it is a black image    
        for idx, part_feat4 in enumerate(behind_down):
            if (part_feat4 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[4][idx] = torch.zeros((512,1,1), dtype=torch.float, device = 'cuda:0')

        # Put zeros if it is a black image    
        for idx, part_feat7 in enumerate(proc_part_feat[7]):
            if (part_feat7 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[5][idx] = torch.zeros((256,1,1), dtype=torch.float, device = 'cuda:0')

        # Put zeros if it is a black image    
        for idx, part_feat8 in enumerate(proc_part_feat[8]):
            if (part_feat8 == torch.zeros((256, 4,4) , dtype=torch.float, device = 'cuda:0')).all():
                new_part_feat[6][idx] = torch.zeros((256,1,1), dtype=torch.float, device = 'cuda:0')

        paw_feature = torch.cat((new_part_feat[1], new_part_feat[2], new_part_feat[3], new_part_feat[4], new_part_feat[5],new_part_feat[6]), dim=1)
        paw_feature = paw_feature.view(paw_feature.shape[0], -1)  
        return paw_feature
        