
#numpy
import numpy as np
#math
import math
#panda
import pandas as pd
#data clean (regular expression and natural language toolkit)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download()

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
#warning
import warnings
#seaborn
import seaborn as sns
#matplot
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

"""
read user csv file from file location and return a dataframe
Input: file location
Output: data frame
"""
def read_user_csv(file_location):
    df = pd.read_csv(file_location, sep = ',', encoding = 'latin-1', usecols = lambda col: col not in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    return df



"""
preprocessing data frame
Input: data frame
Output: data frame upsample
"""
## hint: use resample from sklearn.utils
from sklearn.utils import resample
def preprocessing_dataset(df):
    df_majority = df[df['sentiment'] == 'positive']
    df_minority = df[df['sentiment'] == 'negative']
    negative_upsample = resample(df_minority, replace = True, n_samples = df_majority.shape[0],random_state = 101)
    df_upsampled = pd.concat([df_majority, negative_upsample])  # concat two data frames i,e majority class data set and upsampled minority class data set
    df_upsampled = df_upsampled.sample(frac = 1)
    return df_upsampled




"""
clean the words
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 

"""

def clean_review(review):


    
    # Remove HTML markup
    review_cleaned1  = re.sub('<[^>]*>', '', review)
    
        
    # Remove Bracket
    review_cleaned2 = re.sub('\[[^]]*\]', '', review_cleaned1)
    
    # Remove Special Character
    review_cleaned3 = re.sub(r'[^a-zA-z0-9\s]','',review_cleaned2)

    
    review_cleaned4 = []
    # Removing stop words 
    stop_words = set(stopwords.words('english'))

    for w in review_cleaned3.lower().split():
        if w not in stop_words:
            review_cleaned4.append(w)
    
    #join the text in lowercase and stem the text
    ps=nltk.porter.PorterStemmer()
    review_cleaned5= ' '.join([ps.stem(word) for word in review_cleaned4])
    
    #lemmatize
    review_cleaned6= ' '.join([wn.lemmatize(word,'v') for word in re.split('\W+', review_cleaned5)])

    return review_cleaned6


"""
sub-function for bayes
Params:
    frequency: a dictionary with the frequency of each pair (or tuple)
    word: the word to look up
    label: the label corresponding to the word
Return:
    n: the number of times the word with its corresponding label appears.
"""
def find_occurrence(freqs, word, label):

    n = 0
    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

'''
a dictionary total frequency for each words sub-function for bayes
Params:
    output_occurrence: a dictionary that will be used to map each pair to its frequency
    reviews: a list of reviews
    positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1)
Return:
    output: a dictionary mapping each pair to its frequency
'''
def review_counter(output_occurrence, reviews, positive_or_negative):

    ## Steps :
    # define the key, which is the word and label tuple
    # if the key exists in the dictionary, increment the count
    # else, if the key is new, add it to the dictionary and set the count to 1
    
    for label, review in zip(positive_or_negative, reviews):
      split_review = clean_review(review).split()
      for word in split_review:
        pair = (word, label)
        # Your code here
        if pair in output_occurrence:
            output_occurrence[pair] += 1
        else:
            output_occurrence[pair] = 1

   
    return output_occurrence
   











'''
Train the data with naive bayes
Input:
    freqs: dictionary from (word, label) to how often the word appears
    train_x: a list of reviews
    train_y: a list of labels correponding to the reviews (0,1)
Output:
    logprior: the log prior. (equation 3 above)
    loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
'''

def train_naive_bayes(freqs, train_x, train_y):
    

    loglikelihood = {}
    logprior = 0


    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in freqs.keys():
        # if the label is positive (equal to zero)
        if pair[1] == 0:

            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            num_neg += freqs[pair]

    # Calculate num_doc, the number of documents
    num_doc = len(train_y)
    
    # Calculate D_neg, the number of negative documents 
    neg_num_docs = np.sum(train_y)
    # Calculate D_pos, the number of positive documents 
    pos_num_docs = num_doc - neg_num_docs
    




    # Calculate logprior
    logprior = np.log(neg_num_docs) - np.log(pos_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos =  find_occurrence(freqs, word, 0)
        freq_neg = find_occurrence(freqs, word, 1)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (num_pos + V)
        p_w_neg = (freq_neg + 1) / (num_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_neg/p_w_pos)



    return logprior, loglikelihood


'''
Predict with bayes
Params:
    review: a string
    logprior: a number
    loglikelihood: a dictionary of words mapping to numbers
Return:
    total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

'''
def naive_bayes_predict(review, logprior, loglikelihood):

    
      # process the review to get a list of words
    word_l = clean_review(review).split()

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior
    

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]
            
    if(total_prob >=0.5):
        total_prob = 1
    else:
        total_prob =0
        
    return total_prob




def main():
    
    print("\n\n\n\nRun the program sucessfully!!!! \n\n\n\nPlease wait when we training the dataset... (Training the algorithm takes roughly few seconds) !!!!!\n\n\n\n")
    #read from file and get the data frame
    df = read_user_csv("movie_reviews.csv")

    ## 1. preprocess data set
    df_upsampled = preprocessing_dataset(df)

    ## 2. split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_upsampled['review'], df_upsampled['sentiment'], test_size=0.5, random_state=1)

    ## 3. label the data
    output_map = {'positive': 0, 'negative': 1} ## With the use of mapping function, we replace the label in the form of string to an integer.
    y_train = y_train.map(output_map)
    y_test = y_test.map(output_map)
    ## 4. training data set
    freqs = review_counter({}, X_train, y_train)
    logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train)
    print("Train the algorithm sucessfully!!!!\n")
   
    
    while True:
        val = input("Enter your comment to get the predict outcome (Enter Q to quit): \n")

        if val.lower() == "q":
            break
        else:
            outcome = naive_bayes_predict(val, logprior, loglikelihood)
            if outcome == 1:
                print("negative comment")
            else:
                print("positive comment")


if __name__ == "__main__":
    main()


    
