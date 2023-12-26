import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import BertTokenizer, BertModel
import torch

def get_embeddings_bert(df: pd.DataFrame, model_name: str):
    
    lower_case = True if "uncased"== model_name.split('-')[-1] else False
    
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = lower_case)
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
    
    
    embed_df.to_csv(f'data/embeddings_{model_name}.csv', index=False, mode='w')

def nan_to_empty(y: list):
    for i in range(len(y)):
        try:
            if pd.isna(y[i]):
                y[i] = []
        except:
            pass
    return y

def get_sets_mskf(df: pd.DataFrame, col_name: str, y_df: pd.DataFrame):
    X = pd.DataFrame(df['Embedding'])
    y = pd.DataFrame(y_df[col_name])

    y[col_name] = y[col_name].str.split(', ')
    y = y[col_name].values.tolist()
    y = nan_to_empty(y)
    
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
    
def classifier_fit(train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, col_name: str, learner):
    X, y, mskf = get_sets_mskf(train_dataset, col_name, eval_dataset)
    
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

def get_results(bert_models, learners,  embeddings_path, col_name: str, eval_dataset: pd.DataFrame = None):
    results = []
    for bert_model in bert_models:
        train = pd.read_csv(f'{embeddings_path}_{bert_model}.csv')
        test = eval_dataset if eval_dataset is not None else pd.read_csv(f'{embeddings_path}_{bert_model}.csv') 
        
        for learner in learners:
            f1, acc, prec, rec = classifier_fit(train, test, col_name, learner)
            results.append((bert_model, learner.__class__.__name__, f1, acc, prec, rec))
    
    return results
            
def save_results(results, path):
    with open(path, 'w') as file:
        for item in results:
            file.write(f"Bert model: {item[0]}\n")
            file.write(f"Learner: {item[1]}\n")
            file.write("F1-score: " + str(round(item[2], 4)) + "\n")
            file.write("Accuracy: " + str(round(item[3], 4)) + "\n")
            file.write("Precision: " + str(round(item[4], 4)) + "\n")
            file.write("Recall: " + str(round(item[5], 4)) + "\n")
            file.write("\n")
    
if __name__ == '__main__':
    # Datasets paths
    data_path = 'data/final_reviews.csv'
    embeddings_path = 'data/embeddings'
    frank_path = 'data/frank_reviews.csv'
    lauren_path = 'data/lauren_reviews.csv'
    
    # BERT models names
    bert_model_names = ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased']

    # Read datasets
    origin_df = pd.read_csv(data_path)
    eval_dfs = [frank_path, lauren_path]
    
    # Learners
    learner1 = LogisticRegression(max_iter=1000)
    learner2 = RandomForestClassifier(n_estimators=100)
    learner3 = SVC()
    learner4 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    learners = [learner1, learner2, learner3, learner4]
    
    cols = ['Gender']
    for col in cols:
        #save_results(get_results(bert_model_names, learners, embeddings_path, col), f'results/eval_{col}_final_corpus.txt')
        for df_path in eval_dfs:
            eval_df = pd.read_csv(df_path)
            name = df_path.split('/')[-1].split('.')[0]
            save_results(get_results(bert_model_names, learners, embeddings_path, col, eval_df), f'results/eval_{col}_{name}.txt')