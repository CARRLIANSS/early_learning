import os
import sys
from time import time

import numpy as np
import pandas as pd
from libs.davis17_evaluate.evaluation import DAVISEvaluation

def evalutate(opt, epoch):

    # GT root dir
    default_davis_path = '/home/VOS/DATA_ROOT/DAVIS17/train'  # 使用info划分val

    time_start = time()

    set = 'val'
    task = 'semi-supervised'
    results_path = os.path.join(opt.output_dir, opt.valset)

    csv_name_global = f'global_results-{set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{set}.csv'

    # Check if the method has been evaluated before, if so read the results, otherwise compute the results
    csv_name_global_path = os.path.join(opt.output_dir, csv_name_global)
    csv_name_per_sequence_path = os.path.join(opt.output_dir, csv_name_per_sequence)

    print(f'Evaluating sequences for the {task} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=default_davis_path, task=task, gt_set=set)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array(
        [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
         np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {set} ---------------------------\n")
    print(table_g.to_string(index=False))

    # 每隔5个epoch保存一次benchmark数据
    if (epoch + 1) % 5 == 0 or epoch == opt.epoches:
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.3f")
        print(f'Global results saved in {csv_name_global_path}')

    if epoch == opt.epoches:
        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.3f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

        sys.stdout.write(f"\n---------- Per sequence results for {set} ----------\n")
        print(table_seq.to_string(index=False))

    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))

    return g_res[0, 0]






