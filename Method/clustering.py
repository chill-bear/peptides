import pandas as pd
from Bio import pairwise2 as pw2
from sklearn.cluster import AgglomerativeClustering


def generate_similarity(df):
  sequences = list(set(df['sequence'].tolist()))

  res_dict = {}
  for first_seq in sequences:
      for second_seq in sequences:
          global_align = pw2.align.globalxx(first_seq, second_seq)
          seq_length = min(len(first_seq), len(second_seq))
          matches = global_align[0][2]
          percent_match = (matches / seq_length) * 100
          res_dict[first_seq+ "_" + second_seq] = percent_match
  return res_dict 


def clustering(res_dict, N=100):
  df1 = pd.DataFrame(res_dict.items())
  df2 = pd.concat([df1[[1]], df1[0].str.split('_', expand=True)], axis=1)
  df2.columns = ['sim_score', 'seq1', 'seq2']
  similarity_metric = df2.pivot(index='seq1', columns='seq2',values='sim_score')
  model = AgglomerativeClustering(affinity='euclidean', n_clusters=N, linkage='complete').fit(similarity_metric)
  clustered_df = pd.DataFrame(list(zip(list(similarity_metric.index), model.labels_)))
  clustered_df.columns = ['sequence', 'n_cluster']

  return clustered_df