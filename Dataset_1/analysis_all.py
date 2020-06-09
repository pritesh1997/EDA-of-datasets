import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
master = pd.read_csv('master_dataframe.csv')
master = master[(master.funding_round_type == 'venture') | (master.funding_round_type == 'angel') |
                (master.funding_round_type == 'seed') | (master.funding_round_type == 'private_equity')]
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master)
plt.yscale('log')
plt.show

compare = master.pivot_table(values='raised_amount_usd', columns='funding_round_type', aggfunc=[np.median, np.mean])
compare_median = master.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)
master_venture = master[master['funding_round_type'] == 'venture']
compare_country = master_venture.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False)
top_9 = compare_country[:9]
master_venture = master_venture[(master_venture['country_code']=='USA') | (master_venture['country_code']=='GBR') | (master_venture['country_code']=='IND')]
sns.boxplot(x='country_code', y='raised_amount_usd', data=master_venture)
plt.yscale('log')
plt.show()

master_venture.loc[:, 'main_category'] = master_venture['category_list'].apply(lambda x: x.split('|')[0])
master_venture = master_venture.drop('category_list', axis=1)
mapping = pd.read_csv('mapping.csv')
mapping = mapping[~pd.isnull(mapping['category_list'])]
mapping['category_list'] = mapping['category_list'].str.lower()
master_venture['main_category'] = master_venture['main_category'].str.lower()
master_venture[~master_venture['main_category'].isin(mapping['category_list'])]
mapping[~mapping['category_list'].isin(master_venture['main_category'])]
mapping['category_list'] = mapping['category_list'].apply(lambda x: x.replace('0', 'na'))

final_df = pd.merge(master_venture, mapping, left_on='main_category', right_on='category_list', how='inner')
final_df = final_df.drop('category_list', axis=1)

value_vars = final_df.columns[9:18]
id_vars = np.setdiff1d(final_df.columns, value_vars)
long_df = pd.melt(final_df, id_vars=list(id_vars), value_vars=list(value_vars))
long_df = long_df[~(long_df['value'] == 0)]
long_df = long_df.rename(columns={'variable' : 'sector'})
long_df = long_df.drop('value', axis=1)

df = long_df[(long_df['raised_amount_usd'] >= 5000000) & (long_df['raised_amount_usd'] <= 15000000)]
sector_decription = df.groupby(['country_code', 'sector']).raised_amount_usd.agg(['count', 'sum'])

plt.figure(figsize=(16, 14))

plt.subplot(2, 1, 1)
p = sns.barplot(x='sector', y='raised_amount_usd', hue='country_code', data=df, estimator=np.sum)
p.set_xticklabels(p.get_xticklabels(), rotation=30)
plt.title('Total Invested Amount (USD)')

plt.subplot(2, 1, 2)
q = sns.countplot(x='sector', hue='country_code', data=df)
q.set_xticklabels(q.get_xticklabels(), rotation=30)
plt.title('Number of Investments')

plt.show()

