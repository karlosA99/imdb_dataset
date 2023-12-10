import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit

data_path = 'data/final_reviews.csv'

ds = pd.read_csv(data_path)

def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs

def train_test_validation_split(dataset: pd.DataFrame):
    ds_copy = dataset.copy(deep=True)
    ds_copy['Strat'] = ds_copy[['Gender', 'Race']].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    
    uniques_indivs = pd.DataFrame()
    
    uniq, ds_copy = remove_unique_indiv(ds_copy, "Strat")
    uniques_indivs = pd.concat([uniques_indivs, uniq])
    
    train, temp = train_test_split(ds_copy, test_size=0.3, random_state=42, stratify=ds_copy['Strat'])
    
    uniq, temp = remove_unique_indiv(temp, "Strat")
    uniques_indivs = pd.concat([uniques_indivs, uniq])
    
    test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['Strat'])
    
    return train, test, val
    
def split_into_subsets(dataset: pd.DataFrame, subsets_size: list, *cols_to_strat):
    """Returns a list of subsets of the original dataset, stratified by the given columns.
    The subsets are in the same order as the given sizes.
    Args:
        dataset (pd.DataFrame): Original Dataset
        subsets_size (list): List of sizes for each subset, 0 < size < 1, and sum of sizes = 1
        colums_to_stratify : Names of columns to stratify
    """
    if sum(subsets_size) != 1:
        raise ValueError("Sum of sizes must be equal to 1")
    for size in subsets_size:
        if size <= 0 or size >= 1:
            raise ValueError("Size must be between 0 and 1")
    
    ds_copy = dataset.copy(deep=True)
    
    ds_copy['Strat'] = ds_copy[list(cols_to_strat)].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
     
    subsets = []
    remaining_ds = ds_copy.copy(deep=True)
    
    uniques_indivs = pd.DataFrame()
    
    for i in range(len(subsets_size) - 1):
        unique, remaining_ds = remove_unique_indiv(remaining_ds, "Strat")
        uniques_indivs = pd.concat([uniques_indivs, unique])
        subset, remaining_ds = train_test_split(remaining_ds, test_size=subsets_size[i], random_state=42, stratify=remaining_ds['Strat'])
        subsets.append(subset)
        
        if i == len(subsets_size) - 2:
            subsets.append(remaining_ds)
    
    for subset in subsets:
        print(subset.shape)
    
    print(uniques_indivs.shape)
        
    return subsets
        
#split_into_subsets(ds, [0.7, 0.15, 0.15],"Gender", "Race")
tr, ts, vl = train_test_validation_split(ds)

print(tr)
print("***********************************")
print(ts)
print("***********************************")
print(vl)