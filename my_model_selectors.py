import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        #less is better with BIC
        min_score = float("inf")
        min_model = None

        #get the logarithm of the number of data points
        logN = np.log(len(self.X))

        #get the features needed to determine the number of free parameters
        features = len(self.X[0])
        
        #get the BIC score for each number of components
        for n in range(self.min_n_components, self.max_n_components +1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                #calculate free parameters https://ai-nd.slack.com/archives/C4GQUB39T/p1491489096823164
                p = n**2 + 2*features*n-1

                #BIC = -2 * logL + p * logN
                score = -2*logL + p*logN

                #find the minimum score and model
                if score < min_score:
                    min_score = score
                    min_model = model
            except:
                pass

        return min_model
                
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = float("-inf")
        max_model = None

        for n in range(self.min_n_components, self.max_n_components +1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                #(1/M-1)*SUM(logL of all other words) which is just the average of scores for all other words
                other_words_mean = np.mean([model.score(*self.hwords[word]) for word in self.words if word is not self.this_word])

                #DIC = logL (of this word) - MEAN(logL of all other words)
                score = logL - other_words_mean

                #find the maximum score and model
                if score > max_score:
                    max_score = score
                    max_model = model
            except:
                pass

        return max_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        
        max_score = float("-inf")
        max_model = None

        number_of_splits = 2

        #Experimenting with different number of splits/folds resulted in faster training but less accurate recognition
        #3 splits seemed to be optimal for recognition
        #splitting via each sequence increased the time far too drastically
        #https://www.google.co.za/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=0ahUKEwjeltztvNPTAhUlCMAKHQ2kC-sQtwIIaDAN&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DTIgfjmp-4BA&usg=AFQjCNE4Exuvh9hlKIUBHtQLVCv4wb7b7g&sig2=iGwWzrlzDGTHrCoINIInCA
        if len(self.sequences) > 2:
            number_of_splits = 3

        #create the KFold obj
        kf = KFold(n_splits = number_of_splits)

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                #score for each split
                for cv_train_idx, cv_test_idx in kf.split(self.sequences):
                    try:

                        #get the test and test lengths
                        test, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                        #get the base model
                        model = self.base_model(n)

                        #score the model against the test
                        score = model.score(test, test_lengths)

                        #find the max score and model
                        if score > max_score:
                            max_score = score
                            max_model = model
                    except:
                        pass
            except:
                #some sequences have only 1 sequence long and therefore kf.split will throw an error
                try:                       
                    #model
                    model = self.base_model(n)

                    #get the score of the word
                    score = model.score(self.X, self.lengths)

                    #find the max score and model
                    if score > max_score:
                        max_score = score
                        max_model = model
                except:
                    pass

        return max_model
        
