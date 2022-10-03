# -*- coding: utf-8 -*-
import os
import requests
import pickle
import io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

from evaluation_thumos import prop_eval


def run_evaluation(proposal_filename, groundtruth_filename='/home/ngan_uark/tqsang/AEN_BERT/thumos14_test_groundtruth.csv'):
    frm_nums = pickle.load(open("evaluation_thumos/frm_num.pkl", 'rb'))
    rows = prop_eval.pkl2dataframe(frm_nums, 'evaluation_thumos/movie_fps.pkl', proposal_filename)
    aen_results = pd.DataFrame(rows, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])

    # Retrieves and loads Thumos14 test set ground-truth.
    if not os.path.isfile(groundtruth_filename):
        ground_truth_url = ('https://gist.githubusercontent.com/cabaf/'
                            'ed34a35ee4443b435c36de42c4547bd7/raw/'
                            '952f17b9cdc6aa4e6d696315ba75091224f5de97/'
                            'thumos14_test_groundtruth.csv')
        s = requests.get(ground_truth_url).content
        groundtruth = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=' ')
        groundtruth.to_csv(groundtruth_filename)
    else:
        groundtruth = pd.read_csv(groundtruth_filename)
    # Computes recall for different tiou thresholds at a fixed average number of proposals.
    '''
    recall, tiou_thresholds = prop_eval.recall_vs_tiou_thresholds(aen_results, ground_truth,
                                                        nr_proposals=nr_proposals,
                                                        tiou_thresholds=np.linspace(0.5, 1.0, 11))
    recall = np.mean(recall)
    '''
    average_recall, average_nr_proposals = prop_eval.average_recall_vs_nr_proposals(aen_results, groundtruth)

    return average_recall, average_nr_proposals


def evaluate_proposals(cfg, nr_proposals_list=(50, 100, 200, 500, 1000)):
    average_recall, average_nr_proposals = run_evaluation(cfg.DATA.RESULT_PATH)
    f = interp1d(average_nr_proposals, average_recall, axis=0, bounds_error=False, fill_value='extrapolate')

    ar_results = {}
    for nr_prop in nr_proposals_list:
        ar_results[nr_prop] = float(f(nr_prop))
        print("AR@{} is {}\n".format(nr_prop, ar_results[nr_prop]))

    return ar_results[100]


def plot_metric(average_nr_proposals, recalls, labels, colors, linestyles, figure_file):
    fn_size = 25
    plt.figure(num=None, figsize=(30, 10))

    #colors = ['#2CBDFE', '#47DBCD', '#F3A0F2', '#9D2EC5', '#661D98', '#F5B14C']

    def plotting(sub_ax, recs, lbs, lnstls, clrs):
        for idx, rec in enumerate(recs):
            ax.plot(average_nr_proposals, rec, color=clrs[idx],
                    label=lbs[idx],
                    linewidth=6, linestyle=lnstls[idx], marker=None)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', fontsize=fn_size)

        plt.ylabel('Average Recall', fontsize=fn_size)
        plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
        plt.grid(b=True, which="both")
        #plt.ylim([.35, .6])
        plt.setp(ax.get_xticklabels(), fontsize=fn_size)
        plt.setp(ax.get_yticklabels(), fontsize=fn_size)

    ax = plt.subplot(1, 2, 1)
    plotting(
        ax,
        recalls[:3],
        labels[:3],
        linestyles[:3],
        colors[:3]
    )
    ax = plt.subplot(1, 2, 2)
    plotting(
        ax,
        recalls[3:],
        labels[3:],
        linestyles[3:],
        [colors[0]] + colors[3:]
    )

    # plt.show()
    plt.savefig(figure_file, dpi=300)


def main_evaluate_proposals(result_file, nr_proposals_list):
    average_recall, average_nr_proposals = run_evaluation(result_file)
    f = interp1d(average_nr_proposals, average_recall, axis=0, bounds_error=False, fill_value='extrapolate')

    ar_results = []
    for nr_prop in nr_proposals_list:
        ar_results.append(float(f(nr_prop)))
    return ar_results


def main():
    result_dir = 'results/ablation_study/'
    result_files = [
        'full_arch.pkl',
        'act_only.pkl',
        'env_only.pkl',
        'full_arch.pkl',
        'env+hard_attn_only.pkl',
        'env+self_attn_only.pkl',
    ]
    labels = [
        'AEI (actor and environment)',
        'Actor only',
        'Environment only',
        'AEI (main actor selection and feature fusion)',
        'w/o feature fusion',
        'w/o main actor selection',
    ]
    #colors = ['#2f4858', '#55dde0', '#33658a', '#f6ae2d', '#f26419']
    #colors = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00']
    colors = ['tab:red', 'tab:purple', 'tab:green', 'tab:blue', 'tab:orange']
    linestyles = ['-'] * 6

    nr_props = list(range(50, 1000))

    ar_results = [] 
    for res_file in result_files:
        ar_results.append(main_evaluate_proposals(os.path.join(result_dir, res_file), nr_props))
    plot_metric(nr_props, ar_results, labels, colors, linestyles, 'ablation_study.png')


if __name__ == '__main__':
    main()
