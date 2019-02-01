#!/usr/bin/env python3

import numpy as np
import sys
import json

import matplotlib

matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

datasets = ['peru-Xsqr',
            'D1-Xsqr',
            'D3-Xsqr',
            'D5-Xsqr']

def get_num_ops(dataset):
    return int(open('bfast-futhark/{}.ops'.format(dataset.replace('-Xsqr', ''))).read().splitlines()[2])

def get_futhark_results(variant):
    json_file = 'bfast-futhark/indiv-kernels/matrix-inv/matinv-{}.json'.format(variant)
    prog = 'matinv.fut'
    dataset_results = json.load(open(json_file))[prog]['datasets']
    res = []
    for dataset in datasets:
        num_ops = get_num_ops(dataset)
        dataset_runtimes = dataset_results['../../data/{}.in'.format(dataset)]['runtimes']
        mean_s = np.mean(dataset_runtimes)/1e6
        gflops = (num_ops/mean_s)/1e9
        res += [{'runtime': mean_s, 'gflops': gflops}]
    return res

datas=[(get_futhark_results('fast-mem'), '#888888', 'FastMem'),
       (get_futhark_results('glob-mem'), '#000000', 'GlobMem')]

plt.figure(figsize=(6,2))
ax = plt.subplot(111)
plt.tight_layout()
ind = np.arange(len(datasets))
width=0.2
ax.set_axisbelow(True)
plt.grid(axis='y')
for ((data, color, name), i) in zip(datas, range(len(datas))):
    offset = i*width
    ax.bar(ind+offset, map(lambda x: x['gflops'], data),
           width=width,
           color=color,
           align='center',
           label=name)

ymin, ymax = plt.ylim()

yticks = ax.get_yticks()
ydiff = yticks[1]-yticks[0]


ax.set_yticks(np.concatenate((ax.get_yticks(), ax.get_yticks()[1:] - ydiff/2)))
ax.set_xticks(ind+width*(len(datas)-1)/2.0)
ax.set_xticklabels(map(lambda x: x.replace('-Xsqr', ''), datasets))
ymin, ymax = plt.ylim()
for (x, data) in zip(ax.get_xticks(), datas[0][0]):
    ax.text(x, -ymax/4,
            "$%.2fms$" % (data['runtime']*1000),
            ha='center', va='baseline', weight='bold')

ax.legend(loc='upper center', ncol=len(datas), bbox_to_anchor=(0.5, 1.3), framealpha=1)
plt.ylabel('$GFLOPS^{Sp}$')

if len(sys.argv) > 1:
    plt.savefig(sys.argv[1], bbox_inches='tight')
else:
    plt.show()
