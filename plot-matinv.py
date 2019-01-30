#!/usr/bin/env python3

import numpy as np
import sys
import json

import matplotlib

# matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

datasets = ['sahara-Xsqr',
            'peru-Xsqr',
            'd-Xsqr-16384-1024-512',
            'd-Xsqr-32768-512-256',
            'd-Xsqr-65536-256-128']

def get_num_ops(dataset):
    return 1000000000

def get_futhark_results(variant):
    json_file = 'bfast-futhark/indiv-kernels/matrix-inv/matinv{}.json'.format(variant)
    prog = 'matinv{}.fut'.format(variant)
    dataset_results = json.load(open(json_file))[prog]['datasets']
    res = []
    for dataset in datasets:
        num_ops = get_num_ops(dataset)
        dataset_runtimes = dataset_results['../../data/{}.in'.format(dataset)]['runtimes']
        mean_s = np.mean(dataset_runtimes)/1e6
        gflops = (num_ops/mean_s)/1e9
        res += [gflops]
    return res

datas=[(get_futhark_results(''), '#000000', 'All parallelism'),
       (get_futhark_results('-outer'), '#222222', 'Outer parallelism')]

ax = plt.subplot(111)
plt.tight_layout()
ind = np.arange(len(datasets))
width=0.2

for ((data, color, name), i) in zip(datas, range(len(datas))):
    offset = i*width
    ax.bar(ind+offset, data,
           width=width,
           color=color,
           align='center',
           label=name)

ax.set_xticks(ind+width*(len(datas)-1)/2.0)
ax.set_xticklabels(datasets, rotation=-45)
ax.legend(loc='upper center', ncol=len(datas), bbox_to_anchor=(0.5, 1.05), framealpha=1)

if len(sys.argv) > 1:
    plt.savefig(sys.argv[1], bbox_inches='tight')
else:
    plt.show()