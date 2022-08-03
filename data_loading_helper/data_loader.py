import csv
import os

import pandas as pd
from pandas import merge
import py_entitymatching as em

def _get_record(str):
  if str.startswith('"'):
    return str
  elif ',' in str:
    return '"' + str + ""
  else:
    return str

def write_extended_block_file(A, B, block_file, extended_block_file):
  original_columns = A.columns

  A_dict = {row['id'] : row for row in A.to_dict(orient="records")}
  B_dict = {row['id'] : row for row in B.to_dict(orient="records")}

  extended_columns = ['_id'] + ['ltable_' + c for c in original_columns] + \
                     ['rtable_' + c for c in original_columns]

  with open(extended_block_file, 'w') as result_file:
    writer = csv.DictWriter(result_file, extended_columns)
    writer.writeheader()

    with open(block_file, 'r') as t:
      reader = csv.DictReader(t)
      i = 0
      for r in reader:
        recordA = A_dict[int(r['ltable_id'])]
        recordB = B_dict[int(r['rtable_id'])]

        recordAB = {'_id': i}
        for c in original_columns:
          recordAB['ltable_' + c] = recordA[c]
          recordAB['rtable_' + c] = recordB[c]
        writer.writerow(recordAB)
        i += 1


def load_data(left_file_name, right_file_name, label_file_name,
              block_file, blocking_fn, include_self_join=False):
  A = em.read_csv_metadata(left_file_name, key="id", encoding='iso-8859-1')
  B = em.read_csv_metadata(right_file_name, key="id", encoding='iso-8859-1')
  try:
    G = pd.read_csv(label_file_name)
  except:
    G = None

  extended_block_file = block_file.rsplit('/', 1)[0] + '/extended-train.csv'
  if not os.path.isfile(extended_block_file):
    write_extended_block_file(A, B, block_file, extended_block_file)

  # Read the candidate set
  C = em.read_csv_metadata(extended_block_file, key='_id', ltable=A,
                                 rtable=B, fk_ltable='ltable_id',
                                 fk_rtable='rtable_id')
  # C = blocking_fn(A, B)
  if include_self_join:
    C_A = blocking_fn(A, A)
    C_B = blocking_fn(B, B)
    return A, B, G, C, C_A, C_B
  else:
    return A, B, G, C
