#!/usr/bin/env python3

import numpy as np
import sys
import json

import matplotlib

matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

datasets = ['peru',
            'd-16384-1024-512',
            'd-16384-512-256',
            'd-32768-256-128',
            'd-32768-512-256',
            'd-65536-128-64',
            'd-65536-256-128']

def get_num_ops(dataset):
    return int(open('bfast-futhark/{}.ops'.format(dataset)).read().splitlines()[0])

def get_futhark_results(json_file, prog):
    dataset_results = json.load(open(json_file))[prog]['datasets']
    res = []
    for dataset in datasets:
        num_ops = get_num_ops(dataset)
        dataset_runtimes = dataset_results['data/{}.in'.format(dataset)]['runtimes']
        mean_s = np.mean(dataset_runtimes)/1e6
        gflops = (num_ops/mean_s)/1e9
        res += [{'runtime': mean_s, 'gflops': gflops}]
    return res

def get_c_results():
    res = []
    for dataset in datasets:
        num_ops = get_num_ops(dataset)
        dataset_runtime = int(open('bfast-c/{}.runtime'.format(dataset)).read())
        mean_s = dataset_runtime/1e6
        gflops = (num_ops/mean_s)/1e9
        res += [{'runtime': mean_s, 'gflops': gflops}]
    return res

datas=[(get_futhark_results('bfast-futhark/bfast.json', 'bfast.fut'), '#000000', 'Incremental'),
       (get_futhark_results('bfast-futhark/bfast-moderate.json', 'bfast.fut'), '#222222', 'Moderate'),
       (get_futhark_results('bfast-futhark/bfast-unopt.json', 'bfast-unopt.fut'), '#444444', 'Unoptimized'),
       (get_futhark_results('bfast-futhark/bfast-fused.json', 'bfast-fused.fut'), '#666666', 'Fused'),
       (get_c_results(), '#bbbbbb', 'C')]

plt.figure(figsize=(6,2))
ax = plt.subplot(111)
plt.tight_layout()
ind = np.arange(len(datasets))
width=0.18
ax.set_axisbelow(True)
plt.grid(axis='y')
for ((data, color, name), i) in zip(datas, range(len(datas))):
    offset = i*width
    rects = ax.bar(ind+offset, map(lambda x: x['gflops'], data),
                   width=width,
                   color=color,
                   align='center',
                   label=name)

    ymin, ymax = plt.ylim()

    if i == 0:
        for (r, x) in zip(rects, data):
            plt.text(r.get_x()+width*len(datas)/2, -ymax/4,
                     "$%.2fms$" % (x['runtime']*1000),
                     ha='center', va='baseline', weight='bold')

ax.set_xticks(ind+width*(len(datas)-1)/2.0)
ax.set_xticklabels(datasets)
ax.legend(loc='upper center', ncol=len(datas), bbox_to_anchor=(0.5, 1.25), framealpha=1)
plt.ylabel('GFLOPS')

if len(sys.argv) > 1:
    plt.savefig(sys.argv[1], bbox_inches='tight')
else:
    plt.show()
