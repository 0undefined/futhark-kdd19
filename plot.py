#!/usr/bin/env python3

import numpy as np
import sys
import json

import matplotlib

# matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

datasets = ['data/peru.in',
            'data/d-16384-1024-512.in',
            'data/d-16384-512-256.in',
            'data/d-32768-256-128.in',
            'data/d-32768-512-256.in',
            'data/d-65536-128-64.in',
            'data/d-65536-256-128.in']

def get_futhark_results(json_file, prog):
    dataset_results = json.load(open(json_file))[prog]['datasets']
    runtimes = []
    for dataset in datasets:
        dataset_runtimes = dataset_results[dataset]['runtimes']
        runtimes += [np.mean(dataset_runtimes)]
    return runtimes

datas=[(get_futhark_results('bfast-futhark/bfast.json', 'bfast.fut'), '#000000'),
       (get_futhark_results('bfast-futhark/bfast-moderate.json', 'bfast.fut'), '#222222'),
       (get_futhark_results('bfast-futhark/bfast-unopt.json', 'bfast-unopt.fut'), '#444444'),
       (get_futhark_results('bfast-futhark/bfast-fused.json', 'bfast-fused.fut'), '#666666')]

ax = plt.subplot(111)
ind = np.arange(len(datasets))
width=0.2

for ((data, color), i) in zip(datas, range(len(datas))):
        offset = i*width - len(datas)/2*width
        ax.bar(ind+offset, data,
               width=width,
               color=color,
               align='center')

ax.set_xticks(ind)
ax.set_xticklabels(datasets, rotation=-45)
plt.show()
