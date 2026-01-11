import pandas as pd

topics = ['cows','rabbits','polarbears','financialmarkets']

dfs = [pd.read_excel(f'{topic}.xls') for topic in topics]
for df,topic in zip(dfs,topics):
    df['source'] = topic

combined = pd.concat(dfs)
combined.to_csv('WOS_combined.csv',index=False)
