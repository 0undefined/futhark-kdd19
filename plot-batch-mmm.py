#!/usr/bin/env python3

import numpy as np
import sys
import json

import matplotlib

matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

datasets = ['peru',
            'D1',
            'D2',
            'D3',
            'D4',
            'D5',
            'D6']

def get_num_ops(dataset):
    return int(open('bfast-futhark/{}.ops'.format(dataset.replace('-Xsqr', ''))).read().splitlines()[1])

def get_futhark_results(variant):
    json_file = 'bfast-futhark/indiv-kernels/batch-mmm/bmmm-{}.json'.format(variant)
    prog = 'bmmm-{}.fut'.format(variant)
    dataset_results = json.load(open(json_file))[prog]['datasets']
    res = []
    for dataset in datasets:
        num_ops = get_num_ops(dataset)
        dataset_runtimes = dataset_results['../../data/{}.in'.format(dataset)]['runtimes']
        mean_s = np.mean(dataset_runtimes)/1e6
        gflops = (num_ops/mean_s)/1e9
        res += [{'runtime': mean_s, 'gflops': gflops}]
    return res

datas=[(get_futhark_results('regtile'), '#000000', 'Register tiled'),
       (get_futhark_results('blktile'), '#444444', 'Block tiled'),
       (get_futhark_results('unoptim'), '#888888', 'Unoptimized')]

plt.figure(figsize=(6,2))
ax = plt.subplot(111)
plt.tight_layout()
ind = np.arange(len(datasets))
width=0.2
ax.set_axisbelow(True)
plt.grid(axis='y')
for ((data, color, name), i) in zip(datas, range(len(datas))):
    offset = i*width
    rects = ax.bar(ind+offset, map(lambda x: x['gflops'], data),
                   width=width,
                   color=color,
                   align='center',
                   label=name)

ax.set_xticks(ind+width*(len(datas)-1)/2.0)
ax.set_xticklabels(datasets)

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

