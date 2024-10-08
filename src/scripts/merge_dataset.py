import os
import pandas as pd

df_const = pd.read_csv(os.getcwd() + '/data/constructivo/train_v1.tsv', sep='\t')
df_toxi = pd.read_csv(os.getcwd() + '/data/toxico/train_v2.tsv', sep='\t')

df_const.drop(columns=['Unnamed: 0', 'index_comment', 'article_id',  'TOXICIDAD'], inplace=True)
df_toxi.drop(columns=['Unnamed: 0', 'index_comment', 'article_id', 'CONST'], inplace=True)
# df_concat = pd.concat([df_const, df_toxi], axis=1)
df_concat = df_const.merge(df_toxi, on=['id', 'source', 'text'], how='inner')
df_concat.to_csv(os.getcwd() + '/data/const_toxico_train.tsv', sep='\t', index=False)
print('')