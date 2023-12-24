import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import BertTokenizer, BertModel
import torch


data_path = 'data/final_reviews.csv'
embeddings_path = 'data/embeddings.csv'

origin_df = pd.read_csv(data_path)
#embeddings_df = pd.read_csv(embeddings_path)

def get_embeddings_bert(df: pd.DataFrame):
    model_name = 'bert-large-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = BertModel.from_pretrained(model_name)
    
    embed_df = pd.DataFrame()
    embed_df['Gender'] = df['Gender']
    embed_df['Race'] = df['Race']
    
    for idx, row in df.iterrows():
        text = row['Review']
        tokens = tokenizer.encode_plus(text, add_special_tokens = True, return_tensors = 'pt')
        
        if tokens.input_ids.size(1) > tokenizer.model_max_length:
            chunked_tokens = [tokens.input_ids[:, i:i+tokenizer.model_max_length] 
                              for i in range(0, tokens.input_ids.size(1), tokenizer.model_max_length)]
        
            embeddings = []
        
            for chunk in chunked_tokens:
                with torch.no_grad():
                    outputs = model(input_ids = chunk)
                    chunk_embeddings = outputs.pooler_output
                    chunk_embeddings = chunk_embeddings.squeeze().tolist()
                    embeddings.append(chunk_embeddings)
            embedding = torch.tensor(embeddings).mean(dim=0).tolist()
        else:
            with torch.no_grad():
                outputs = model(**tokens)
                embedding = outputs.pooler_output
                embedding = embedding.squeeze().tolist() 
                
        
        
        embed_df.at[idx, 'Embedding'] = str(embedding)
    
    
    embed_df.to_csv('data/embeddings.csv', index=False, mode='w')

def get_sets_mskf(df: pd.DataFrame, col_name: str):
    X = pd.DataFrame(df['Embedding'])
    y = pd.DataFrame(df[col_name])
    y = y.fillna('Null')
    y[col_name] = y[col_name].str.split(', ')
    y = y[col_name].values.tolist()
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    y_transformed = mlb.transform(y)
    
    return X, y_transformed, mskf

def df_to_data_list(df: pd.DataFrame):
    data_list = []
    
    for _, row in df.iterrows():
        embeddings = row['Embedding']
        embeddings = ast.literal_eval(embeddings)
        data_list.append(embeddings)
    data_list = np.array(data_list)
    return data_list
    
def classifier_fit(dataset: pd.DataFrame, col_name: str, learner):
    X, y, mskf = get_sets_mskf(dataset, col_name)
    
    classifier = MultiOutputClassifier(learner)
    
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for train_index, test_index in mskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train = df_to_data_list(X_train)
        X_test = df_to_data_list(X_test)
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        f1_scores.append(f1_score(y_test, y_pred, average='micro'))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='micro'))
        recall_scores.append(recall_score(y_test, y_pred, average='micro'))
    
    return np.mean(f1_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)
        
            
def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs

get_embeddings_bert(origin_df)

learner1 = LogisticRegression(max_iter=1000)
learner2 = RandomForestClassifier(n_estimators=100)
learner3 = SVC()
learner4 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
learners = [learner1, learner2, learner3, learner4]

# for learner in learners:
#     f1, acc, prec, rec = classifier_fit(embeddings_df, 'Gender', learner)
#     print(learner.__class__.__name__)
#     print(f'F1 Score: {f1}', f'Accuracy: {acc}', f'Precision: {prec}', f'Recall: {rec}')
#     print("*********************************************************************************")