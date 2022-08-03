import csv
import fcntl
import os
import resource
import time

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from data_loading_helper.data_loader import load_data
from data_loading_helper.feature_extraction import *
from utils import run_zeroer
import utils
from blocking_functions import *
from os.path import join
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset",type=str)
parser.add_argument("--run_transitivity",type=bool,default=False,nargs="?",const=True, help="whether to enforce transitivity constraint")
parser.add_argument("--LR_dup_free",type=bool,default=False,nargs="?",const=True, help="are the left table and right table duplicate-free?")
parser.add_argument("--LR_identical",type=bool,default=False,nargs="?",const=True, help="are the left table and right table identical?")

data_path = "datasets"

if __name__ == '__main__':
    args = parser.parse_args()
    LR_dup_free = args.LR_dup_free
    run_trans = args.run_transitivity
    LR_identical = args.LR_identical
    dataset_name = args.dataset
    dataset_path = join(data_path,dataset_name)
    blocking_func = blocking_functions_mapping[dataset_name]
    try:
        candset_features_df = pd.read_csv(join(dataset_path,"candset_features_df.csv"), index_col=0)
        candset_features_df.reset_index(drop=True,inplace=True)
        if run_trans==True:
            id_df = candset_features_df[["ltable_id","rtable_id"]]
            id_df.reset_index(drop=True,inplace=True)
            if LR_dup_free==False and LR_identical==False:
                candset_features_df_l = pd.read_csv(join(dataset_path,"candset_features_df_l.csv"), index_col=0)
                candset_features_df_l.reset_index(drop=True,inplace=True)
                candset_features_df_r = pd.read_csv(join(dataset_path,"candset_features_df_r.csv"), index_col=0)
                candset_features_df_r.reset_index(drop=True,inplace=True)
                id_df_l = candset_features_df_l[["ltable_id","rtable_id"]]
                id_df_l.reset_index(drop=True,inplace=True)
                id_df_r = candset_features_df_r[["ltable_id","rtable_id"]]
                id_df_r.reset_index(drop=True,inplace=True)
        print(
            "Features already generated, reading from file: " + dataset_path + "/candset_features_df.csv")

    except FileNotFoundError:
        print("Generating features and storing in: " + dataset_path + "/candset_features_df.csv")

        f = open(join(dataset_path, 'metadata.txt'), "r")
        LEFT_FILE = join(dataset_path, f.readline().strip())
        if LR_identical:
            RIGHT_FILE = LEFT_FILE
        else:
            RIGHT_FILE = join(dataset_path, f.readline().strip())
        DUPLICATE_TUPLES = join(dataset_path, f.readline().strip())
        BLOCK_FILE = join(dataset_path, 'train.csv')
        f.close()
        if run_trans==True and LR_dup_free==False and LR_identical==False:
            ltable_df, rtable_df, duplicates_df, candset_df,candset_df_l,candset_df_r = load_data(LEFT_FILE, RIGHT_FILE, DUPLICATE_TUPLES,
                                                                                              BLOCK_FILE, blocking_func,
                                                                                              include_self_join=True)
        else:
            ltable_df, rtable_df, duplicates_df, candset_df = load_data(LEFT_FILE, RIGHT_FILE, DUPLICATE_TUPLES,
                                                                                              BLOCK_FILE, blocking_func,
                                                                                              include_self_join=False)
            if LR_identical:
                print("removing self matches")
                candset_df = candset_df.loc[candset_df.ltable_id!=candset_df.rtable_id,:]
                candset_df.reset_index(inplace=True,drop=True)
                candset_df['_id'] = candset_df.index
        if duplicates_df is None:
            duplicates_df = pd.DataFrame(columns=["ltable_id", "rtable_id"])
        candset_features_df = gather_features_and_labels(ltable_df, rtable_df, duplicates_df, candset_df)
        candset_features_df.to_csv(join(dataset_path,"candset_features_df.csv"))
        id_df = candset_df[["ltable_id", "rtable_id"]]

        if run_trans == True and LR_dup_free == False and LR_identical==False:
            duplicates_df_r = pd.DataFrame()
            duplicates_df_r['l_id'] = rtable_df["id"]
            duplicates_df_r['r_id'] = rtable_df["id"]
            candset_features_df_r = gather_features_and_labels(rtable_df, rtable_df, duplicates_df_r, candset_df_r)
            candset_features_df_r.to_csv(join(dataset_path,"candset_features_df_r.csv"))


            duplicates_df_l = pd.DataFrame()
            duplicates_df_l['l_id'] = ltable_df["id"]
            duplicates_df_l['r_id'] = ltable_df["id"]
            candset_features_df_l = gather_features_and_labels(ltable_df, ltable_df, duplicates_df_l, candset_df_l)
            candset_features_df_l.to_csv(join(dataset_path,"candset_features_df_l.csv"))

            id_df_l = candset_df_l[["ltable_id","rtable_id"]]
            id_df_r = candset_df_r[["ltable_id","rtable_id"]]
            id_df_l.to_csv(join(dataset_path,"id_tuple_df_l.csv"))
            id_df_r.to_csv(join(dataset_path,"id_tuple_df_r.csv"))

    similarity_features_df = gather_similarity_features(candset_features_df)
    similarity_features_lr = (None,None)
    id_dfs = (None, None, None)
    if run_trans == True:
        id_dfs = (id_df, None, None)
        if LR_dup_free == False and LR_identical==False:
            similarity_features_df_l = gather_similarity_features(candset_features_df_l)
            similarity_features_df_r = gather_similarity_features(candset_features_df_r)
            features = set(similarity_features_df.columns)
            features = features.intersection(set(similarity_features_df_l.columns))
            features = features.intersection(set(similarity_features_df_r.columns))
            features = sorted(list(features))
            similarity_features_df = similarity_features_df[features]
            similarity_features_df_l = similarity_features_df_l[features]
            similarity_features_df_r = similarity_features_df_r[features]
            similarity_features_lr = (similarity_features_df_l,similarity_features_df_r)
            id_dfs = (id_df, id_df_l, id_df_r)

    true_labels = candset_features_df.gold.values
    if np.sum(true_labels)==0:
        true_labels = None

    start_time = time.time()
    y_pred = run_zeroer(similarity_features_df, similarity_features_lr,id_dfs,
                        true_labels ,LR_dup_free,LR_identical,run_trans)
    end_time = time.time()
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pred_df = candset_features_df[["ltable_id","rtable_id"]]
    pred_df['pred'] = y_pred
    pred_df.to_csv(join(dataset_path,"pred.csv"))

    # Appending results
    predicted_labels = np.round(np.clip(y_pred + utils.DEL, 0., 1.)).astype(int)
    p = precision_score(true_labels, predicted_labels)
    r = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    f_star = 0 if (p + r - p * r) == 0 else p * r / (p + r - p * r)

    result_file = '/home/remote/u6852937/projects/results.csv'
    file_exists = os.path.isfile(result_file)

    with open(result_file, 'a') as results_file:
        heading_list = ['method', 'dataset_name', 'train_time', 'test_time',
                        'train_max_mem', 'test_max_mem', 'TP', 'FP', 'FN',
                        'TN', 'Pre', 'Re', 'F1', 'Fstar']
        writer = csv.DictWriter(results_file, fieldnames=heading_list)

        if not file_exists:
          writer.writeheader()

        fcntl.flock(results_file, fcntl.LOCK_EX)
        result_dict = {
            'method' : 'zeroer',
            'dataset_name': dataset_name,
            'train_time': end_time - start_time,
            'test_time': 'NA',
            'train_max_mem': max_mem,
            'test_max_mem': 'NA',
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Pre': round(p * 100, 2),
            'Re': round(r * 100, 2),
            'F1': round(f1 * 100, 2),
            'Fstar': round(f_star * 100, 2)
        }
        writer.writerow(result_dict)
        fcntl.flock(results_file, fcntl.LOCK_UN)

