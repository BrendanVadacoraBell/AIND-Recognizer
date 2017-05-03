import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    #https://discussions.udacity.com/t/recognizer-implementation/234793/5

    for x, lengths in test_set.get_all_Xlengths().values(): 

        sequence_word_dict = {} #dict for word probabilities at each iteration
        best_score = float("-inf")
        best_word = None

        for word, model in models.items():
            #calculate the scores for each model(word) and update the 'probabilities' list.
            score = float("-inf")
            try:
                score = model.score(x, lengths)
            except:
                pass

            sequence_word_dict[word] = score

            #determine the maximum score for each model(word).
            if best_score < score:
                best_score = score
                best_word = word

        #append the dictionary for each iteration to the probabilites list
        probabilities.append(sequence_word_dict)
        #add the best guessed word to the guesses list
        guesses.append(best_word)
        
    return probabilities, guesses
    
