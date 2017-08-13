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


def seq_to_mat_2D(SeqArr1, SeqArr2):

    '''
    Convert two length-m RNA sequence into a 9*m matrix.
    (1) The central nucleotide whose rank is smaller on an RNA sequence 
        will be placed above. The sequence placed below is reversed.
    (2) The 5th row indicate the possible pairing partners and there 
        neighborhoods that the network need to decide.
    '''

    if(len(SeqArr1) != len(SeqArr2)):
        print("Error in seq_to_mat_2D! Two input sequences has different lengths.")
        
    SeqDict = { 'A' : np.array([1,0,0,0], dtype="float32"),
                'C' : np.array([0,1,0,0], dtype="float32"),
                'G' : np.array([0,0,1,0], dtype="float32"),
                'U' : np.array([0,0,0,1], dtype="float32"),
                'T' : np.array([0,0,0,1], dtype="float32"),
                'N' : np.array([0,0,0,0], dtype="float32")}

    data = np.zeros((9,len(SeqArr1)), dtype="float32")
    for i in range(len(SeqArr1)):
        data[0:4,i]      = SeqDict[SeqArr1[i]]
        data[5:9,-(i+1)] = SeqDict[SeqArr2[i]]

    md_num = (len(SeqArr1)-1)/2
    data[4, md_num] = 1.0
    if ((SeqArr1[md_num+1] != 'N') and (SeqArr2[md_num-1] != 'N')):
        data[4, md_num+1] = 0.5
    if ((SeqArr1[md_num+2] != 'N') and (SeqArr2[md_num-2] != 'N')):
        data[4, md_num+2] = 0.25
    if ((SeqArr1[md_num-1] != 'N') and (SeqArr2[md_num+1] != 'N')):
        data[4, md_num-1] = 0.5
    if ((SeqArr1[md_num-2] != 'N') and (SeqArr2[md_num+2] != 'N')):
        data[4, md_num-2] = 0.25

    return data


def fill_window(j, SeqArr, Winsize):
    
    '''
    Place the RNA sequence into a fixed window. All the empty cells
    are fullfilled by Ns.
    '''
    
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


def prep_data_2D(Winsize, dir_path):
    #Prepare the training data and labels accroding to the Window size
    extension_list = ['ct']
    seqlist = get_file_list(dir_path, extension_list)

    Data = []
    Pair = []
    No_pair = []
    seqnum = 0

    for file in seqlist:
        fh = open(file,'r')
        headline = fh.readline()
        Headline = headline.strip().split()
        Headline[0] = int(Headline[0])
        if Headline[0] < Winsize:
            SeqArr = []
            Pairbase = []
            Pairdict = {}
            for line in fh.readlines():
                Arr = []
                Arr=line.strip().split()
                SeqArr.append(Arr[1])
                Arr[4]=int(Arr[4])
                Arr[0]=int(Arr[0])
                if(Arr[4]!=0):
                    Pairbase.append(Arr[0])
                    if(Arr[4]-Arr[0]>0):
                        Pairdict[Arr[0]] = Arr[4]
                        Pair.append([seqnum, Arr[0], Arr[4]]) #The order of nucleotides is 1 based!
                        
            Data.append(SeqArr) # All sequences are stored in 'Data'.
            pairnum = len(Pairbase)
            for i in range(pairnum-1):
                for j in range(i+1, pairnum):
                    if Pairbase[i] in Pairdict:
                        if (Pairbase[j] != Pairdict[Pairbase[i]]):
                            No_pair.append([seqnum, Pairbase[i], Pairbase[j]])
                    else:
                        No_pair.append([seqnum, Pairbase[i], Pairbase[j]])
            seqnum += 1
    
    No_pair_new = []
    for i in range(len(No_pair)):
        if(   (Data[No_pair[i][0]][No_pair[i][1]-1]=="A" and Data[No_pair[i][0]][No_pair[i][2]-1]=="U")
           or (Data[No_pair[i][0]][No_pair[i][1]-1]=="C" and Data[No_pair[i][0]][No_pair[i][2]-1]=="G")
           or (Data[No_pair[i][0]][No_pair[i][1]-1]=="G" and Data[No_pair[i][0]][No_pair[i][2]-1]=="C")
           or (Data[No_pair[i][0]][No_pair[i][1]-1]=="G" and Data[No_pair[i][0]][No_pair[i][2]-1]=="U")
           or (Data[No_pair[i][0]][No_pair[i][1]-1]=="U" and Data[No_pair[i][0]][No_pair[i][2]-1]=="A")
           or (Data[No_pair[i][0]][No_pair[i][1]-1]=="U" and Data[No_pair[i][0]][No_pair[i][2]-1]=="G")):
            No_pair_new.append(No_pair[i])

    random.shuffle(No_pair_new)
    return No_pair_new, Pair, Data


def prep_data_2D_iterative(Winsize, j, No_pair, Pair, Data):

    num=len(Pair)
    Pos_Arr = np.zeros((num,9,Winsize), dtype="float32")
    Neg_Arr = np.zeros((num,9,Winsize), dtype="float32")

    for i in range(num):
        SeqArr1 = fill_window(Pair[i][1]-1, Data[Pair[i][0]], Winsize)
        SeqArr2 = fill_window(Pair[i][2]-1, Data[Pair[i][0]], Winsize)
        Pos_Arr[i] = seq_to_mat_2D(SeqArr1, SeqArr2)

    for i in range(num):
        SeqArr1 = fill_window(No_pair[i+j*num][1]-1, Data[No_pair[i+j*num][0]], Winsize)
        SeqArr2 = fill_window(No_pair[i+j*num][2]-1, Data[No_pair[i+j*num][0]], Winsize)
        Neg_Arr[i] = seq_to_mat_2D(SeqArr1, SeqArr2)

    np.random.shuffle(Neg_Arr)
    np.random.shuffle(Pos_Arr)    
    training_data = np.zeros((num*2,9,Winsize), dtype="float32")
    training_label= np.zeros((num*2), dtype="int8")
    n_pos = 0
    n_neg = 0

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

    return training_data, training_label


def load_test(Winsize, No_pair, Pair, Data):

    Pos_Arr = np.zeros((len(Pair),9,Winsize), dtype="float32")
    Neg_Arr = np.zeros((len(No_pair),9,Winsize), dtype="float32")
    
    for i in range(len(Pair)):
        SeqArr1 = fill_window(Pair[i][1]-1, Data[Pair[i][0]], Winsize)
        SeqArr2 = fill_window(Pair[i][2]-1, Data[Pair[i][0]], Winsize)
        Pos_Arr[i] = seq_to_mat_2D(SeqArr1, SeqArr2)
    
    for i in range(len(No_pair)):
        SeqArr1 = fill_window(No_pair[i][1]-1, Data[No_pair[i][0]], Winsize)
        SeqArr2 = fill_window(No_pair[i][2]-1, Data[No_pair[i][0]], Winsize)
        Neg_Arr[i] = seq_to_mat_2D(SeqArr1, SeqArr2)

    X_pos = Pos_Arr
    y_pos = [1 for i in range(len(Pair))]
    X_neg = Neg_Arr
    y_neg = [0 for i in range(len(No_pair))]

    return X_pos, y_pos, X_neg, y_neg
