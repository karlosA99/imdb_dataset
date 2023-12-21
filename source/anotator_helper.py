
from random import randint
import pandas as pd
import shutil
from numpy import nan

def int_to_gender(number): 
    switcher = {
        '1': "Male",
        '2': "Female",
        '3': "Male, Female",
        '4': "Null"}
    return switcher.get(number, nan)

def int_to_race(number):
    switcher = {
        '1': "Black",
        '2': "White",
        '3': "Asian",
        '4': "Latino",
        '5': 'Native American',
        '6': 'Indian',
        '7': 'Arab',
        '8': "Null"}
    return switcher.get(number, nan)
    
def list_to_races(numbers):
    numbers = list(set(numbers.split(',')))
    races_str = ''
    
    if len(numbers) > 1:
        for n in numbers:
            races_str += int_to_race(n) + ', '
    else:
        return int_to_race(numbers[0])
    
    return races_str[:-1]

#Primero conformamos el nuevo dataset, con los 70 reviews ya anotados y 80 mas.    
def add_reviews_from_to(source, dest, random_choice, *new_cols):
    """Agrega nuevos reviews desde source hacia dest.

    Args:
        source (csv file): Fuente de datos que se utilizara para llenar dest.
        dest (csv file): Donde se almacenaran los datos extraidos de source.
        random_choice (int): Determina la cantidad de datos random que se pasaran de source a dest. En caso de ser 0, se 
        pasan todos los datos.
        *new_cols (str): son columnas nuevas para agregar a dest.
    """
    old_reviews = pd.read_csv(source)
    try:
        extended_reviews = pd.read_csv(dest)
    except:
        
        old_cols = list(old_reviews.columns)
        cols_names = old_cols[:-1] + list(new_cols) + [old_cols[-1]]
        
        extended_reviews = pd.DataFrame(columns=cols_names)
        
    #copiar todo el dataset
    if not random_choice:
        for col in old_reviews:
            extended_reviews[col] = old_reviews[col]
    else:
        count_revs = random_choice
        while count_revs > 0:
            rnd = randint(0, old_reviews.shape[0] - 1)
            review = old_reviews['Review'].iloc[rnd]
            sentiment = old_reviews['Sentiment'].iloc[rnd]
            
            if sentiment == 0:
                sentiment = 'negative'
            if sentiment == 1:
                sentiment = 'positive' 
                        
            if review not in extended_reviews['Review'].values:
                count_revs -= 1
                new_row = pd.DataFrame({'Review': [review], 'Sentiment': [sentiment]})
                extended_reviews = pd.concat([extended_reviews, new_row], ignore_index=True)
    
    extended_reviews.to_csv(dest, index=False)
            
# add_reviews_from_to('data/reviews.csv', 'data/extended_reviews.csv', 0, 'Race')
# add_reviews_from_to('data/train_data.csv', 'data/extended_reviews.csv', 80)

# Ahora hay que empezar manualmente lo que falta en el dataset. Para ello se utiliza el siguiente codigo:
def run_annotations():
    annotator_code = input('Ingrese su codigo de anotador: ')
    reviews_path = 'data/extended_reviews.csv'

    try:
        annotator_reviews = pd.read_csv(f'data/{annotator_code}_reviews.csv')
    except:
        shutil.copy2(reviews_path, f'data/{annotator_code}_reviews.csv')
        annotator_reviews = pd.read_csv(f'data/{annotator_code}_reviews.csv')
    
    for idx, row in annotator_reviews.iterrows():
        
        if pd.isna(annotator_reviews.loc[idx, 'Race']):
            print('Seleccione la(s) raza(s) que se aprecia(n) en el siguiente review:\n')
            print(row['Review'] + '\n')
            print('1. Negro, 2. Blanco, 3. Asiatico, 4. Latino, 5. Nativo Americano, 6. Indio, 7. Arabe, 8. No se puede determinar\n')
            print('En caso de determinar la raza de un personaje de ser necesario se debe realizar una busqueda en internet.\n')
        

            races = input('Ingrese los numeros correspondientes a las razas separados por comas en caso de ser necesario. Ejemplo: 1,2,3\n')
            print('\n')
            races = list_to_races(races)
            
            annotator_reviews.at[idx, 'Race'] = races
            annotator_reviews.to_csv(f'data/{annotator_code}_reviews.csv', index=False, mode='w')
            
            
        if pd.isna(annotator_reviews.loc[idx, 'Gender']):
            #En este caso no esta definido el genero, por lo que hay que anotarlo
            print('Seleccione el genero que se aprecia en el siguiente review: \n')
            print(row['Review'] + '\n')
            print('1. Solo Masculino, 2. Solo Femenino, 3. Ambos, 4. No se puede determinar\n')
            gender = int_to_gender(input('Ingrese el numero correspondiente: \n'))
            print('\n')
            annotator_reviews.at[idx, 'Gender'] = gender
            annotator_reviews.to_csv(f'data/{annotator_code}_reviews.csv', index=False, mode='w')
        
run_annotations()