import pandas as pd
import shutil

empty_revs_path = 'data/extended_reviews.csv'
final_revs_path = 'data/final_reviews.csv'

def equal_decisions(datasets, idx, col):
    are_equals = True
    decisions = []
    size = len(datasets)
    
    for i in range(size):
        decisions.append(datasets[i].loc[idx, col])
        
        if datasets[i%size].loc[idx, col] != datasets[(i+1)%size].loc[idx, col]:
            are_equals = False
            
    return are_equals, decisions

def sort_race(path):
    dataframe = pd.read_csv(path)
    
    for idx, row in dataframe.iterrows():
        item = dataframe.loc[idx, 'Race']
        item = item.split(',')
        item.sort()
        item = ','.join(item)
        dataframe.at[idx, 'Race'] = item
    
    dataframe.to_csv(path, index=False, mode='w')

def get_final_decision(decisions, review):
    print("Ingrese el numero de la opcion que considere mas acertada:")
    print("Review:")
    print(review)
    print("=====================================")
    print("Decisions:")
    
    for i in range(len(decisions)):
        print(str(i+1) + ": " + decisions[i])
        
    print("=====================================")
    choice = input('Choice:')
    
    try:
        choice = int(choice)-1
        return decisions[choice]
    except:
        return None
    
def merge(*paths):
    try:
        final_reviews = pd.read_csv(f'data/final_reviews.csv')
    except:
        shutil.copy2(empty_revs_path, final_revs_path)
        final_reviews = pd.read_csv(final_revs_path)
    
    datasets = [pd.read_csv(path) for path in paths]
    
    for idx, row in final_reviews.iterrows():
        if pd.isna(final_reviews.loc[idx, 'Gender']):
            are_equals, decisions = equal_decisions(datasets, idx, 'Gender')
            if not are_equals:
                final_decision = get_final_decision(decisions, final_reviews.loc[idx, 'Review'])
                final_reviews.at[idx, 'Gender'] = final_decision
            else:
                final_reviews.at[idx, 'Gender'] = decisions[0]
            
            final_reviews.to_csv(final_revs_path, index=False, mode='w')
            
            
        if pd.isna(final_reviews.loc[idx, 'Race']):
            are_equals, decisions = equal_decisions(datasets, idx, 'Race')
            if not are_equals:
                final_decision = get_final_decision(decisions, final_reviews.loc[idx, 'Review'])
                final_reviews.at[idx, 'Race'] = final_decision
            else:
                final_reviews.at[idx, 'Race'] = decisions[0]
            
            final_reviews.to_csv(final_revs_path, index=False, mode='w')

merge('data/lauren_reviews.csv', 'data/frank_reviews.csv')
