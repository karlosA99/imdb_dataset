import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MultiLabelBinarizer


data_path = 'data/final_reviews.csv'

df = pd.read_csv(data_path)

df_fill = df.fillna('Null')
df_fill['Gender'] = df_fill['Gender'].str.split(', ')
df_fill['Race'] = df_fill['Race'].str.split(',')

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = df_fill.drop(columns='Gender')
y = df_fill['Gender']
print(y)
y = y.values.tolist()
print(y)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
print(y)

for train_index, test_index in mskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train)
print("********************************")
print(X_test)

def replace_mf(df_fill: pd.DataFrame):
    
    for idx, row in df_fill.iterrows():
        if df_fill.loc[idx, 'Gender'] == 'Male, Female':
            df_fill.at[idx, 'Gender'] = [1,1,0]
            
        elif df_fill.loc[idx, 'Gender'] == 'Male':
            df_fill.at[idx, 'Gender'] = [1,0,0]
            
        elif df_fill.loc[idx, 'Gender'] == 'Female':
            df_fill.at[idx, 'Gender'] = [0,1,0]
            
        elif df_fill.loc[idx, 'Gender'] == 'Null':
            df_fill.at[idx, 'Gender'] = [0,0,1]
        
def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs

#Not Multilabel
def train_test_validation_split(dataset: pd.DataFrame, col_name : str):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    data_removed, dataset = remove_unique_indiv(dataset, col_name)
    
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
    
    return X_train, test_set, X_val, data_removed

#Multilabel
def train_test_val_split(dataset: pd.DataFrame, col_name: str):
    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print(type_of_target(dataset[col_name]))
    
    train_indices, test_indices = next(mskf.split(dataset, dataset[col_name]))
    
    train_set = dataset.iloc[train_indices].copy()
    test_set = dataset.iloc[test_indices].copy()
    
    train_indices, val_indices = next(mskf.split(train_set, train_set[col_name]))
    
    train_set = train_set.iloc[train_indices].copy()
    val_set = train_set.iloc[val_indices].copy()
    
    return train_set, test_set, val_set
    


    