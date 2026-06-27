#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:


from collections import namedtuple
import sys
# sys.path.append(r"D:\Lung_Cancer_Project")


# In[ ]:


CandidateInfoTuple = namedtuple("CandidateInfoTuple","isNodule_bool diameter_mm series_uid center_xyz")
# these tuples are just for sanitizing our data


# In[ ]:


import os
# os.getcwd()


# In[ ]:


# pip uninstall diskcache


# In[ ]:


# pip install diskcache==4.1.0


# In[ ]:


import diskcache
# print(diskcache.__version__)


# In[ ]:


# get_ipython().system('pip install SimpleITK')


# In[ ]:

import math
import random
import functools
import copy
import csv
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.cuda
from torch.utils.data import Dataset
from util.logconf import logging
from util.disk import getCache
from util.util import XyzTuple, xyz2irc
import SimpleITK as sitk

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG) # this helps in giving constructive error messages if something occurs

@functools.lru_cache(1) # stores the value of below function in the cache and gives it out instead of running the function again if the input is the same

def getCandidateInfoList(requireOnDisk_bool = True):
    '''this function creates the candidate info list with centres and diameter of nodules along with series uid'''
    mhd_list = glob.glob("D:/Lung_Cancer_Project/subset*/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open("D:/Lung_Cancer_Project/annotations.csv","r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(p) for p in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid,[]).append((annotationCenter_xyz,annotationDiameter_mm))



    candidateInfo_list = []


    with open("D:/Lung_Cancer_Project/candidates.csv","r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            if series_uid not in presentOnDisk_set and requireOnDisk_bool :
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(p) for p in row[1:4]])
            candidateDiameter_mm = 0.00
            # now we are comparing the locations of nodules as per the annotations and the candiadates sheet

            for annotation_tuple in diameter_dict.get(series_uid,[]):
                annotationCenter_xyz,annotationDiameter_mm = annotation_tuple
                for i in range(3):
                    if abs(annotationCenter_xyz[i] - candidateCenter_xyz[i]) > annotationDiameter_mm/4 :
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz
            ))
        candidateInfo_list.sort(reverse=True) # sorted based on diameter
        
        return candidateInfo_list
        
              


# In[ ]:


from diskcache import FanoutCache
cache_path = os.path.join("D:/Lung_Cancer_Project", "cache")
raw_cache = FanoutCache(cache_path, shards=64, timeout=1)
class Ct:
    def __init__(self,series_uid):
        mhd_path = glob.glob(f"D:/Lung_Cancer_Project/subset*/{series_uid}.mhd")[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd),dtype = np.float32)

        ct_a.clip(-1000,1000,ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.VxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)


    def getRawCandidate(self, center_xyz, width_irc):
        ''' this function is used to get a 3d chunk of the ct-scan '''
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.VxSize_xyz,
            self.direction_a
        )

        slice_list = []
        for axis,center_val in enumerate(center_irc):
            start_ind = int(round(center_val - width_irc[axis]/2))
            end_ind = int(start_ind + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid,center_xyz,self.origin_xyz,self.VxSize_xyz,center_irc,axis])
            if start_ind < 0:
                start_ind = 0
                end_ind = width_irc[axis]
            elif end_ind > self.hu_a.shape[axis]:
                end_ind = self.hu_a.shape[axis]
                start_ind = end_ind - width_irc[axis]

            slice_list.append(slice(start_ind,end_ind))
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk,center_irc


@functools.lru_cache(1, typed=True) # caches the most recent value of ct
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)  # caches all the values of the function parameters 
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


def getCtAugmentedCandidate(
    augmentation_dict,
    series_uid,center_xyz,width_irc,
    use_cache = True):

    if use_cache :
        ct_chunk,center_irc = getCtRawCandidate(series_uid,center_xyz,width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)


    ct_t = torch.from_numpy(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32) # unsqueezing 2 times as it is needed for affine_grid ,grid_sample function, but we will squeeze it once in get_item before passing it to the dataloader

    transform_t = torch.eye(4)

    for i in range(3):
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype = torch.float32)

        transform_t @= rotation_t
    
    affine_t = nn.functional.affine_grid( # tells that from which voxel in the domain the given voxel in the range came from
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners = False
    )
    augmented_chunk = nn.functional.grid_sample(
        ct_t,
        affine_t,
        padding_mode = "border",
        align_corners = False
    ).to("cpu")

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk) # this creates a tesor of the same size as the given tensor with random normalised values
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0] ,center_irc # we gave augmented chuck[0], why [0] ?

class LunaDataset(Dataset):
    def __init__(
        self,
        series_uid = None,
        isValSet_bool = None,
        sortby_str = "random",
        val_stride = None,
        ratio_int = 0,
        augmentation_dict = None,
        candidateInfo_list = None,

    ):
        # self.candidateInfo_list = (getCandidateInfoList())
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else :
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True


        

        if series_uid != None:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if series_uid == x.series_uid
            ]
        if isValSet_bool :
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0 :
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list



        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))


        self.negative_list = [x for x in self.candidateInfo_list if x.isNodule_bool == False]
        self.positive_list = [x for x in self.candidateInfo_list if x.isNodule_bool == True]

        log.info("{!r} : {} {} samples , {} pos , {} neg , {} ratio ".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.positive_list),
            len(self.negative_list),
            "{}:1".format(self.ratio_int) if self.ratio_int else "unbalanced" 
            )   
        )
        
    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)

    def __len__(self):
        if self.ratio_int:
            return 200000
        return len(self.candidateInfo_list)

    def __getitem__(self,ind):

        if self.ratio_int:
            pos_idx = ind//(self.ratio_int+1)
            if ind%(self.ratio_int+1):
                neg_idx = ind - 1 - pos_idx
                neg_idx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_idx]
            else:
                pos_idx %= len(self.positive_list)
                candidateInfo_tup = self.positive_list[pos_idx]

        else:
            candidateInfo_tup = self.candidateInfo_list[ind]

                
        width_irc = (32,48,48)

        '''before augmentation code'''
        # candidate_a, center_irc = candidateInfo_tup.series_uid,candidateInfo_tup.center_xyz,width_irc)
        # candidate_t = torch.from_numpy(candidate_a)
        # candidate_t = candidate_t.to(torch.float32)
        # candidate_t = candidate_t.unsqueeze(0)
        
        '''after augmentation code'''
        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)


        '''as nn.crossEntropyLoss expects a one output value per class'''
        pos_t = torch.tensor([ 
            not candidateInfo_tup.isNodule_bool ,
            candidateInfo_tup.isNodule_bool  
            ],
                             
            dtype = torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )

        """
        pipeline :

        raw CT
        ↓
        cache
        ↓
        augmentation
        ↓
        tensor
        ↓
        Dataset return
        ↓
        DataLoader batching

        """