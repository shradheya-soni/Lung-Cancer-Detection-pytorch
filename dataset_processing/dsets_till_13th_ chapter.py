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

# Luna2dSegmentationDataset
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
import scipy.ndimage.morphology as morph
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

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')
raw_cache = getCache('ch13_raw')

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG) # this helps in giving constructive error messages if something occurs

@functools.lru_cache(1) # stores the value of below function in the cache and gives it out instead of running the function again if the input is the same

def getCandidateInfoList(requireOnDisk_bool = True):
    '''this function creates the candidate info list with centres and diameter of nodules along with series uid'''
    mhd_list = glob.glob("D:/Lung_Cancer_Project/subset*/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []

    with open("D:/Lung_Cancer_Project/annotations_with_malignancy.csv","r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(p) for p in row[1:4]])
            annotationDiameter_mm = float(row[4])

            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )




    with open("D:/Lung_Cancer_Project/candidates.csv","r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            if series_uid not in presentOnDisk_set and requireOnDisk_bool :
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(p) for p in row[1:4]])
            
            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz
                    )
                )
        candidateInfo_list.sort(reverse=True) # sorted based on diameter
        
        return candidateInfo_list
 # we have stored the suspects which are either non-nodules or malignent nodules
 # i.e. we have not yet considered the nodules which are benign       
              
@functools.lru_cache(1)
def getCandidateInfoDict(requiredOnDisk_bool = True):
    candidateInfo_list = getCandidateInfoList(requiredOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,[]).append(candidateInfo_tup)

    return candidateInfo_dict
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

        
        CandidateInfo_list=  getCandidateInfoDict()[series_uid]

        self.positiveInfo_list = [
            candidate_tup 
            for candidate_tup in CandidateInfo_list
            if candidate_tup.isNodule_bool
        ]

        # building a mask for positive nodule
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist()) 
        # we are taking the sum of positive pixels for each Series_uid
        # how many pixels in this Series_uid belong to a nodule
    
    def buildAnnotationMask(self,positiveInfo_list,threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a,dtype=bool)
        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                    candidateInfo_tup.center_xyz,
                    self.origin_xyz,
                    self.VxSize_xyz,
                    self.direction_a,
                )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2

            try:
                while self.hu_a[ci + index_radius, cr,cc] > threshold_hu and\
                        self.hu_a[ci - index_radius, cr,cc] > threshold_hu:
                    index_radius += 1
            except IndexError :
                index_radius -= 1

            row_radius = 2

            try:
                while self.hu_a[ci , cr + row_radius,cc] > threshold_hu and\
                        self.hu_a[ci , cr - row_radius,cc] > threshold_hu:
                    row_radius += 1
            except IndexError :
                row_radius -= 1

            col_radius = 2

            try:
                while self.hu_a[ci , cr,cc + col_radius] > threshold_hu and\
                        self.hu_a[ci , cr,cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError :
                col_radius -= 1
            
            assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            assert row_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            assert col_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            
            boundingBox_a[
                ci - index_radius : ci + index_radius+1,
                cr - row_radius : cr + row_radius +1,
                cc - col_radius : cc + col_radius +1
            ] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)
        return mask_a


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

            if end_ind > self.hu_a.shape[axis]:
                end_ind = self.hu_a.shape[axis]
                start_ind = end_ind - width_irc[axis]

            slice_list.append(slice(start_ind,end_ind))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk,pos_chunk,center_irc


@functools.lru_cache(1, typed=True) # caches the most recent value of ct
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)  # caches all the values of the function parameters 
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk,pos_chunk,center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_chunk.clip(-1000,1000, ct_chunk)
    return ct_chunk,pos_chunk,center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes



class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
    ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid] # for a single sample
        else :
            self.series_list = sorted(getCandidateInfoDict().keys()) # for all the series_uid in the candidate info list


        '''previously we did the training using the nodule data indivisually now we are using the whole ct scan of a person'''

        
        if isValSet_bool:
            assert val_stride > 0 ,val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)
            #index-cnt = number of slices
            #positive-indexes = number of positive slices

            if self.fullCt_bool: # whether we are taking the full ct or the cropped up parts 
                self.sample_list += [(series_uid,slice_ndx) for slice_ndx in range(index_count)]
            else:    
                self.sample_list += [(series_uid,slice_ndx) for slice_ndx in positive_indexes]
        
        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list) # copied this for quick access
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self,ndx):
            
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)] #just to make sure that ndx doesnot exceed len of sample list
        return self.getitem_fullSlice(series_uid, slice_ndx)
        
    def getitem_fullSlice(self,series_uid,slice_ndx):
        ct = getCt(series_uid)
            # we want to get nodule prediction at currect slice
            #but it may not be very clear just from one slice ,so we use some slice above and below : contextSlices
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
        ct_t.clamp_(-1000,1000)
        ct_t/=1000
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0).to(torch.float32)  
        return ct_t, pos_t, ct.series_uid,slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ratio_int = 2
    
    def __len__(self):
        return 300000
    
    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)
    
    def __getitem__(self,ndx):
        candidateInfo_tup = self.pos_list[ndx%len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self,candidateInfo_tup):
        # these are in 3d
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        # we have 7 slices : -3 -2 -1 0 1 2 3 , so we took the middle one ie. the 3rd index
        pos_a = torch.from_numpy(pos_a[3]).unsqueeze(0) # size becomes (1,96,96) 2d

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)

        ct_t = torch.from_numpy(ct_a[:,row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        ct_t.clamp_(-1000, 1000)
        ct_t /= 1000

        pos_t = pos_a[:,row_offset:row_offset+64,
                                     col_offset:col_offset+64].to(torch.float32)
        
        slice_ndx = center_irc.index
        
        return ct_t,pos_t,candidateInfo_tup.series_uid,slice_ndx
