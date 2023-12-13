import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

data_path = 'data/final_reviews.csv'

ds = pd.read_csv(data_path)

def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs

def train_test_validation_split(dataset: pd.DataFrame, col_name : str):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    X = dataset.drop(col_name, axis=1)
    y = dataset[col_name]
    
    for train_index, test_index in skf.split(X, y):
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
        
        train_X = train_set.drop(col_name, axis=1)
        train_y = train_set[col_name]
        X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.15, random_state=42, stratify=train_y)
    
    
    
    X_train[col_name] = y_train
    X_val[col_name] = y_val
    test_set.loc[:, col_name] = dataset.loc[test_index, col_name]
    
    print("Train set: ", X_train.shape)
    print("Validation set: ", X_val.shape)
    print("Test set: ", test_set.shape)
    
    return X_train, test_set, X_val
    
#train, test, val = train_test_validation_split(ds, 'Gender')

    