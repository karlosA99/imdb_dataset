import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import BertTokenizer, BertModel
import torch
from copy import deepcopy


class Model:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        
    
    def remove_value_from(self, col_name: str, value: str):
        new_df = pd.DataFrame()
        i = 0
        for idx, row in self.dataframe.iterrows():
            if self.dataframe.at[idx, col_name] != value:
                new_df.at[i, col_name] = self.dataframe.at[idx, col_name]
                i+=1
        self.dataframe = new_df 
    
    def predict(self, indexs, binarized_y):
        predictions = []
        
        for idx in indexs:
            pred = binarized_y[idx]
            predictions.append(pred)
        
        return predictions

class FrankModel(Model):
    pass
class LaurenModel(Model):
    pass

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

def get_sets_mskf(df: pd.DataFrame, col_name: str):
    X = pd.DataFrame(df['Embedding'])
    y = pd.DataFrame(df[col_name])
    
    # Remove only sample with race Native American from the dataset
    if col_name == 'Race':
        for idx, _ in y.iterrows():
            if y.at[idx, col_name] == 'Native American':
                X = X.drop(idx)
                y = y.drop(idx)
                break
    
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

def macro_accuracy(y_true, y_pred):
    y_pred = list(y_pred)
    y_true = list(y_true)
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)
    for i in range(len(y_true)):
        y_true[i] = [j+1 for j in range(len(y_true[i])) if y_true[i][j]]
        y_pred[i] = [j+1 for j in range(len(y_pred[i])) if y_pred[i][j]]
        #y_true[i][j] = j+1 if y_true[i][j] else 0
        #y_pred[i][j] = j+1 if y_pred[i][j] else 0
            
        
    ac_counter = {}

    for true_ann, pred_ann in zip(y_true, y_pred):
        true_ann = frozenset(true_ann)
        pred_ann = frozenset(pred_ann)
        
        correct, total = ac_counter.get(true_ann, (0, 0))
        equal = int(true_ann == pred_ann)
        ac_counter[true_ann] = (correct + equal, total + 1)
    
    total_correct = 0
    total_total = 0
    total_accuracy = 0
    
    for _, (correct, total) in ac_counter.items():
        total_correct += correct
        total_total += total
        total_accuracy += correct/total if total else 0
    
    return total_accuracy/len(ac_counter) if len(ac_counter) else 0
    
# def macro_accuracy(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
    
#     classes = y_true.shape[1]
#     classes_accuracies = []
#     count_cls_correct = []
    
#     for cls in range(classes):
#         for value in range(y_true.shape[0]):
#             if y_true[value][cls] == y_pred[value][cls]:
#                 count_cls_correct.append(cls)
    
#     count_cls_correct = Counter(count_cls_correct)
    
#     for cls in count_cls_correct.keys():
#         classes_accuracies.append(count_cls_correct[cls]/y_true.shape[0])
    
#     return np.mean(classes_accuracies)

def classifier_fit(train_dataset: pd.DataFrame, col_name: str, learner):
    X, y, mskf = get_sets_mskf(train_dataset, col_name)
    
    if isinstance(learner, Model):
        classifier = deepcopy(learner)
        
        if col_name == 'Race':
            classifier.remove_value_from("Race", "Native American")
        
        classifier.dataframe[col_name] = classifier.dataframe[col_name].str.split(', ')
        classifier_y = classifier.dataframe[col_name].values.tolist()
        classifier_y = nan_to_empty(classifier_y)
        
        mlb = MultiLabelBinarizer()
        mlb.fit(classifier_y)
        classifier_y = mlb.transform(classifier_y)
    else: 
        classifier = MultiOutputClassifier(deepcopy(learner))
    
    accuracy_scores = []
    macro_accuracy_scores = []
    micro_f1_scores = []
    macro_f1_scores = []
    micro_precision_scores = []
    macro_precision_scores = []
    micro_recall_scores = []
    macro_recall_scores = []
    
    for train_index, test_index in mskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if isinstance(learner, Model):    
            y_pred = classifier.predict(test_index, classifier_y)
        else:
        
            X_train = df_to_data_list(X_train)
            X_test = df_to_data_list(X_test)
        
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        # Micro scores    
        micro_f1_scores.append(f1_score(y_test, y_pred, average='micro'))
        micro_precision_scores.append(precision_score(y_test, y_pred, average='micro'))
        micro_recall_scores.append(recall_score(y_test, y_pred, average='micro'))
        
        # Macro scores
        macro_accuracy_scores.append(macro_accuracy(y_test, y_pred))
        macro_f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        macro_precision_scores.append(precision_score(y_test, y_pred, average='macro'))
        macro_recall_scores.append(recall_score(y_test, y_pred, average='macro'))
        
    to_return = ((np.mean(accuracy_scores), np.mean(macro_accuracy_scores)), 
                (np.mean(micro_f1_scores), np.mean(macro_f1_scores)), 
                (np.mean(micro_precision_scores), np.mean(macro_precision_scores)), 
                (np.mean(micro_recall_scores), np.mean(macro_recall_scores)))
    return to_return
                   
def remove_unique_indiv(dataset: pd.DataFrame, col_name: str):
    dupli_rows = dataset.duplicated(subset=[col_name], keep=False)
    unique_indivs = dataset[~dupli_rows]
    non_unique_indivs = dataset[dupli_rows]
    
    return unique_indivs, non_unique_indivs

def get_results(bert_models, learners,  embeddings_path, col_name: str):
    results = []
    for bert_model in bert_models:
        train = pd.read_csv(f'{embeddings_path}_{bert_model}.csv')
                
        for learner in learners:
            acc, f1, prec, rec = classifier_fit(train, col_name, learner)
            results.append((bert_model, learner.__class__.__name__, acc, f1, prec, rec))
    
    return results
            
def save_results(results, path):
    with open(path, 'w') as file:
        for item in results:
            file.write(f"Bert model: {item[0]} - ")
            file.write(f"Learner: {item[1]}\n")
            file.write("Accuracy: " + str(round(item[2][0], 4)) + "\n")
            file.write("Macro_Accuracy: " + str(round(item[2][1], 4)) + "\n")
            file.write("Micro_F1-score: " + str(round(item[3][0], 4)) + "\n")
            file.write("Macro_F1-score: " + str(round(item[3][1], 4)) + "\n")
            file.write("Micro_Precision: " + str(round(item[4][0], 4)) + "\n")
            file.write("Macro_Precision: " + str(round(item[4][1], 4)) + "\n")
            file.write("Micro_Recall: " + str(round(item[5][0], 4)) + "\n")
            file.write("Macro_Recall: " + str(round(item[5][1], 4)) + "\n")
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
    frank_df = pd.read_csv(frank_path)
    lauren_df = pd.read_csv(lauren_path)
    
    # Learners
    learner1 = LogisticRegression(max_iter=1000)
    learner2 = RandomForestClassifier(n_estimators=100)
    learner3 = SVC()
    learner4 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    frank_model = FrankModel(frank_df)
    lauren_model = LaurenModel(lauren_df)
    learners = [frank_model, lauren_model, learner1, learner2, learner3, learner4]
    
    cols = ['Gender', 'Race']
    for col in cols:
        save_results(get_results(bert_model_names, learners, embeddings_path, col), f'results/eval_{col}.txt')