import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


data_path = 'data/final_reviews.csv'

df = pd.read_csv(data_path)


def mskf_split_train_test(dataset: pd.DataFrame, col_name: str):
    df_fill = dataset.fillna('Null')
    df_fill['Gender'] = df_fill['Gender'].str.split(', ')
    df_fill['Race'] = df_fill['Race'].str.split(',')
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    X = df_fill.drop(columns=col_name)
    y = df_fill[col_name].values.tolist()
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    
    for train_index, test_index in mskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    return X_train, y_train, X_test, y_test
            
def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs
