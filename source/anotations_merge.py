import pandas as pd

empty_revs_path = 'data/extended_reviews.csv'

def equal_decisions(datasets, idx, col):
    for i in range(len(datasets)-1):
        if datasets[i].loc[idx, col] != datasets[i+1].loc[idx, col]:
            return False
    return True

def sort_race(path):
    dataframe = pd.read_csv(path)
    
    for idx, row in dataframe.iterrows():
        item = dataframe.loc[idx, 'Race']
        item = item.split(',')
        item.sort()
        item = ','.join(item)
        dataframe.at[idx, 'Race'] = item
    
    dataframe.to_csv(path, index=False, mode='w')
    
def merge(*paths):
    final_reviews = pd.read_csv(empty_revs_path)
    
    datasets = [pd.read_csv(path) for path in paths]
    
    for idx, row in final_reviews.iterrows():
        if pd.isna(final_reviews.loc[idx, 'Gender']):
            if not equal_decisions(datasets, idx, 'Gender'):
                #aqui hay que mostrar el review y las anotaciones y que el experto escoja la mas acertada
                pass
            else:
                final_reviews.at[idx, 'Gender'] = datasets[0].loc[idx, 'Gender']
            
        if pd.isna(final_reviews.loc[idx, 'Race']):
            if not equal_decisions(datasets, idx, 'Race'):
                #aqui hay que mostrar el review y las anotaciones y que el experto escoja la mas acertada
                pass
            else:
                final_reviews.at[idx, 'Race'] = datasets[0].loc[idx, 'Race']
