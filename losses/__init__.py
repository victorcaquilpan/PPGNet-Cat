# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss

def make_loss(parameters, num_classes):                                                
    sampler = parameters['DATALOADER_SAMPLER']
    if parameters['METRIC_LOSS_TYPE'] == 'triplet':
        triplet = TripletLoss(parameters['SOLVER_MARGIN'])  # triplet loss
    elif parameters['METRIC_LOSS_TYPE'] == 'cluster':
        cluster = ClusterLoss(parameters['CLUSTER_MARGIN'], True, True, parameters['IMS_PER_BATCH'] // parameters['DATALOADER_NUM_INSTANCE'], parameters['DATALOADER_NUM_INSTANCE'])
    elif parameters['METRIC_LOSS_TYPE'] == 'triplet_cluster':
        triplet = TripletLoss(parameters['SOLVER_MARGIN'])  # triplet loss
        cluster = ClusterLoss(parameters['CLUSTER_MARGIN'], True, True, parameters['IMS_PER_BATCH'] // parameters['DATALOADER_NUM_INSTANCE'], parameters['DATALOADER_NUM_INSTANCE'])
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(parameters['METRIC_LOSS_TYPE']))

    if parameters['IF_LABELSMOOTH'] == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif parameters['DATALOADER_SAMPLER'] == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif parameters['DATALOADER_SAMPLER'] == 'softmax_triplet':
        def loss_func(score_g, score_gb, score_gp, g_b_feature, g_p_feature, target):   #<-----------------用的是这个
            if parameters['METRIC_LOSS_TYPE'] == 'triplet':
                if parameters['IF_LABELSMOOTH'] == 'on':
                    return xent(score_g, target), xent(score_gb, target), xent(score_gp, target), \
                           triplet(g_b_feature, target)[0], triplet(g_p_feature, target)[0]  # new add by luo, open label smooth
                else:
                    pass
                    # return F.cross_entropy(score, target) + triplet(feat, target)[0]    # new add by luo, no label smooth

            elif parameters['METRIC_LOSS_TYPE'] == 'cluster':
                if parameters['IF_LABELSMOOTH'] == 'on':
                    return xent(score, target) + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + cluster(feat, target)[0]    # new add by luo, no label smooth

            elif parameters['METRIC_LOSS_TYPE'] == 'triplet_cluster':
                if parameters['IF_LABELSMOOTH'] == 'on':
                    return xent(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]    # new add by luo, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(parameters['METRIC_LOSS_TYPE']))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(parameters['DATALOADER_SAMPLER']))
    return loss_func

# Call the function with the default parameters from PPGNet
def make_loss_with_parameters(num_classes):

    # Main parameters
    PARAMETERS = {
        'DATALOADER_SAMPLER' : 'softmax_triplet',
        'METRIC_LOSS_TYPE' : 'triplet',
        'SOLVER_MARGIN' : 0.3,
        'CLUSTER_MARGIN' : 0.3,
        'IMS_PER_BATCH' : 32,
        'DATALOADER_NUM_INSTANCE' : 4,
        'IF_LABELSMOOTH': 'on',
        'MODEL_NAME' : 'resnet152',
        'SOLVER_RANGE_K' : 2,
        'SOLVER_RANGE_MARGIN': 0.3,
        'SOLVER_RANGE_ALPHA': 0,
        'SOLVER_RANGE_BETA': 1,
        'SOLVER_CENTER_LOSS_WEIGHT': 0.0005,
        'SOLVER_RANGE_LOSS_WEIGHT': 1}
    
    loss_func = make_loss(PARAMETERS,num_classes=num_classes)
    return loss_func
