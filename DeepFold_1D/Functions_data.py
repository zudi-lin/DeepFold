#!/usr/bin/env python

import numpy as np
import os,glob
import random

def get_file_list(dir_path, extension_list):
    '''
        fuction: get_file_list(dir_path,extension_list)
        parms:
        dir_path : a string of directory full path. eg. 'user/scientist'
        extension_list : a list of file extension. eg. ['ct']
        The function returns a list of file full path.
    '''
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension) ]
    return file_list

def seq_to_mat(SeqArr):
    '''
    Convert an length-m RNA sequence in to a 4*m matrix.
    
    e.g. "ACGUAN" ->  [1 0 0 0 1 0
                       0 1 0 0 0 0
                       0 0 1 0 0 0
                       0 0 0 1 0 0]
    '''
    SeqDict = { 'A' : np.array([1,0,0,0], dtype="float32"),
                'C' : np.array([0,1,0,0], dtype="float32"),
                'G' : np.array([0,0,1,0], dtype="float32"),
                'U' : np.array([0,0,0,1], dtype="float32"),
                'T' : np.array([0,0,0,1], dtype="float32"),
                'N' : np.array([0,0,0,0], dtype="float32")}

    data = np.zeros((4, len(SeqArr)), dtype="float32")
    for i in range(len(SeqArr)):
        data[:,i] = SeqDict[SeqArr[i]]

    #Process the 4*m matrix into a 6*m matrix due to the adding of other features.
    data_more = np.zeros((6,len(SeqArr)), dtype="float32")
    data_more[0:4,:] = data
    md_num = (len(SeqArr)-1)/2
    data_more[4,md_num] = 1.0

    # Check if the neighborhoods are not empty.
    if (SeqArr[md_num+1] != 'N'):
        data_more[4, md_num+1] = 0.5
    if (SeqArr[md_num+2] != 'N'):
        data_more[4, md_num+2] = 0.25
    if (SeqArr[md_num-1] != 'N'):
        data_more[4, md_num-1] = 0.5
    if (SeqArr[md_num-2] != 'N'):
        data_more[4, md_num-2] = 0.25

    # Indicate all possible pairing patterners.
    if (SeqArr[md_num] == 'A'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'U' or SeqArr[index] == 'T'):
                data_more[5, index] = 1.0
                
    if (SeqArr[md_num] == 'C'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'G'):
                data_more[5, index] = 1.0

    if (SeqArr[md_num] == 'G'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'C' or (SeqArr[index] == 'U' or SeqArr[index] == 'T')):
                data_more[5, index] = 1.0

    if (SeqArr[md_num] == 'U' or SeqArr[md_num] == 'T'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'G' or SeqArr[index] == 'A'):
                data_more[5, index] = 1.0

    return data_more


def fill_window(j, SeqArr, Winsize):
    newSeq = ['N']*Winsize
    mi = int((Winsize-1)/2)
    Seqlen = len(SeqArr)

    if Seqlen-j>mi+1: #Exceed the right end of the window
        newSeq[mi-j:mi]=SeqArr[0:j]
        newSeq[mi:Winsize]=SeqArr[j:j+mi+1]
        newSeq[0:Seqlen-(j+mi+1)] =SeqArr[j+mi+1:Seqlen]
    else:
        if j>mi:      #Exceed the left end of the window
            newSeq[mi:mi+Seqlen-j]=SeqArr[j:Seqlen]
            newSeq[0:mi]=SeqArr[j-mi:j]
            newSeq[Winsize-(j-mi):Winsize] =SeqArr[0:j-mi]
        else:    
            newSeq[mi-j:mi+Seqlen-j]=SeqArr[0:Seqlen]

    return newSeq


def prep_data(Winsize, dir_path):
    #Prepare the training data and labels accroding to the Window size
    extension_list = ['ct']
    seqlist = get_file_list(dir_path, extension_list)

    Arr = []
    Data = []
    label = []
    Pos = 0
    Neg = 0

    for file in seqlist:
        fh = open(file,'r')
        headline = fh.readline()
        Headline = headline.strip().split()
        Headline[0] = int(Headline[0])
        if Headline[0] < Winsize:
            '''
            The window size is always an odd, for the purpose of symmetry.

            The length of the RNA sequence that can be loaded into the "Training Box"
            is at most (Winsize-1), because at least one empty cell should exit to re-
            present the differences between the first nucleotide and the last one. 
            '''
            SeqArr = []
            for line in fh.readlines():
                Arr=line.strip().split()
                SeqArr.append(Arr[1])
                Arr[4]=int(Arr[4])
                if(Arr[4]!=0):
                    Pos += 1
                    label.append(1)
                else:
                    Neg += 1
                    label.append(0)
                #Finish the labeling of training samples
            Data.append(SeqArr)
        
    num = max(Pos,Neg)
    '''
        Convert the sequence into a 6-channel data structure:
        (1) Channel 0-3 the four-dimensional vector representation of the sequence.
        (2) Channel 4 indicates the target nucleotide and its neighborhoods: [..., 0.25, 0.5, 1, 0.5, 0.25, ...]
        (3) Channel 5 indicate all other nucleotides on the sequence that can pair with the target one
            according to the canonnical pairing rule.
    '''    
    Pos_Arr = np.empty((Pos,6,Winsize), dtype="float32")
    Neg_Arr = np.empty((Neg,6,Winsize), dtype="float32")
    n_pos=0
    n_neg=0
    k=0 # k is the index of array "label"

    for i in range(len(Data)):
        for j in range(len(Data[i])):
            Arr = fill_window(j, Data[i], Winsize)
            if label[k]==0:
                Neg_Arr[n_neg] = seq_to_mat(Arr)
                n_neg += 1
            elif label[k]==1:
                Pos_Arr[n_pos] = seq_to_mat(Arr)
                n_pos += 1
            k += 1

    np.random.shuffle(Neg_Arr)
    np.random.shuffle(Pos_Arr)    
    training_data = np.zeros((num*2,6,Winsize), dtype="float32")
    training_label= np.zeros((num*2,1), dtype="int8")
    n_pos=0
    n_neg=0

    ran_list=random.sample(range(num*2), num)
    for i in ran_list:
        training_data[i] = Pos_Arr[n_pos]
        training_label[i]= 1
        n_pos += 1
    
    All = [i for i in range(num*2)]
    ran_list=list(set(All).difference(set(ran_list)))
    for i in ran_list:
        training_data[i] = Neg_Arr[n_neg]
        training_label[i]= 0
        n_neg += 1
        if (n_neg==len(Neg_Arr)):
            n_neg = 0

    return training_data, training_label
