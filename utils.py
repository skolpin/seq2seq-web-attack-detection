# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:34:20 2021

@author: Kolpinsky.Ser
"""
import numpy as np
import re

HTTP_RE = re.compile(r"ST@RT.+?INFO\s+(.+?)\s+END", re.MULTILINE | re.DOTALL)

def http_re(data):
    """
    Extracts HTTP requests from raw data string in special logging format.
    Logging format `ST@RT\n%(asctime)s %(levelname)-8s\n%(message)s\nEND`
    where `message` is a required HTTP request bytes.
    """
    return HTTP_RE.findall(data)

def get_requests_from_file(path):
    """
    Reads raw HTTP requests from given file.
    """
    with open(path, 'r') as f:
        file_data = f.read()
    requests = http_re(file_data)
    return requests

def strlist_to_ohearray(requests, max_seq_len, token_index, decoder = False, target = False):
    num_tokens = len(token_index)
    
    if decoder:
        indexarray = np.zeros((len(requests), max_seq_len+2), dtype = 'int16')
        indexarray[:, 0] = token_index['<GO>']
        for i, request in enumerate(requests):
            for t, char in enumerate(request[:max_seq_len]):
                try:
                    indexarray[i, t+1] = token_index[char]
                except KeyError:
                    indexarray[i, t+1] = token_index['<UNK>']
            indexarray[i, t + 2 : -1] = token_index['<PAD>']
        indexarray[:, -1] = token_index['<EOR>']
    else:    
        indexarray = np.zeros((len(requests), max_seq_len), dtype = 'int16')
        for i, request in enumerate(requests):
            for t, char in enumerate(request[:max_seq_len]):
                try:
                    indexarray[i, t] = token_index[char]
                except KeyError:
                    indexarray[i, t] = token_index['<UNK>']
            indexarray[i, t + 1 :] = token_index['<PAD>']    
    
    if target:
        indexarray[:, :-1] = indexarray[:, 1:]
        indexarray[:, -1] = token_index['<PAD>']
    
    result = np.zeros(indexarray.shape + tuple([num_tokens]), dtype = "int8")
    for i in range(indexarray.shape[0]):
        for j in range(indexarray.shape[1]):
            result[i,j,indexarray[i,j]] = 1
    
    return result
