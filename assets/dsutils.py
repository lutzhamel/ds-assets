'''
sklearnutils.py
A collection of utility functions for sklearn models.
(c) University of Rhode Island - Lutz Hamel
'''

import os
import subprocess

import sklearn.utils
if 'google.colab' in os.sys.modules:
  subprocess.run(['pip3','install','PyMySQL'])

# requires
# pip install pymysql
# pip install seaborn
# pip install statistics
# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install scikit-learn

import warnings
import math
import pandas as pd
import statistics as stats
import numpy
import sklearn
import matplotlib.pyplot as plt
import pymysql as sql
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

#################################################################################
# model evaluation

def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    if acc > 1.0 or acc < 0.0:
      raise ValueError('Expected an accuracy value between 0 and 1')
    # if acc == 1.0 pertub the acc for the lower
    # bound so we get something reasonable in terms of interval
    if acc == 1.0:
      lbacc = 0.99
    else:
      lbacc = acc
    lb = max(0, acc - 1.96*math.sqrt(lbacc*(1-lbacc)/n))
    ub = min(1.0, acc + 1.96*math.sqrt(acc*(1-acc)/n))
    return (lb,ub)

def regression_confint(rs_score, n, k):
    '''
    Compute the 95% confidence interval for a regression problem.
      rs_score -- R^2 score
      n        -- number of observations used to compute the R^2 score
      k        -- number of independent variables in dataset
    Returns a tuple (lb,ub)

    Reference:
    https://books.google.com/books?id=gkalyqTMXNEC&pg=PA88#v=onepage&q&f=false
    '''
    interval = 2*math.sqrt((4*rs_score*(1-rs_score)**2*(n-k-1)**2)/((n**2 - 1)*(n+3)))
    lb = max(0, rs_score - interval)
    ub = min(1.0, rs_score + interval)
    return (lb,ub)

def acc_score(model, X, y, as_string=False):
    '''
    Compute the accuracy score for a classification model together
    with its 95% confidence interval.
    Parameters:
      model -- a classification model
      X     -- sklearn style feature matrix
      y     -- sklearn style target vector
      as_string -- if True return a formatted string, otherwise
                   return a tuple (accuracy score, lower bound, upper bound)
    Returns (accuracy score,lower bound, upper bound)
    '''
    acc = model.score(X,y)
    lb,ub = classification_confint(acc, X.shape[0])

    if as_string:
       return f"Accuracy: {acc:.2f} ({lb:.2f}, {ub:.2f})"
    else:
       # return as a tuple
       # (accuracy score, lower bound, upper bound)
       return (acc, lb, ub)

def rs_score(model, X, y, as_string=False):
    '''
    Compute the R^2 score for a regression model together
    with its 95% confidence interval.
    Parameters:
      model -- a regression model
      X     -- sklearn style feature matrix
      y     -- sklearn style target vector
      as_string -- if True return a formatted string, otherwise
                   return a tuple (R^2 score, lower bound, upper bound)
    Returns  (R^2 score, lower bound, upper bound)
    ''' 
    rs = model.score(X,y)
    lb,ub = regression_confint(rs, X.shape[0], X.shape[1])

    if as_string:
       return f"R^2 Score: {rs:.2f} ({lb:.2f}, {ub:.2f})"
    else:
       # return as a tuple
       # (r^2 score, lower bound, upper bound)
       return (rs, lb, ub)

def bootstrap_score(model, X, y, s=200, as_string=False):
    '''
    Compute a bootstrapped model score together with its 95% probability 
    bound. If the model object is a classification model then model accuracy 
    is computed and if the model object is a regression model then the R^2 
    score is computed.

    Parameters
        model - a classification/regression model
        X - sklearn style feature matrix
        y - sklearn style target vector
        s - number of bootstrap samples
        as_string -- if True return a formatted string, otherwise
                   return a tuple (score, lower bound, upper bound)
   
    Returns
        (score, lower bound, upper bound) 
    '''
    score_list = []
    for i in range(s):
        BX,By = sklearn.utils.resample(X,y)
        bootmodel = sklearn.base.clone(model).fit(BX, By)
        score = bootmodel.score(BX,By)
        score_list.append(score)
    score_avg = stats.mean(score_list)
    score_list.sort()
    score_ub = numpy.percentile(score_list,97.5)
    score_lb = numpy.percentile(score_list,2.5)
    if as_string:
       return f"Score: {score_avg:.2f} ({score_lb:.2f}, {score_ub:.2f})"
    else:
       # return as a tuple
       # (score, lower bound, upper bound)
      return (score_avg, score_lb, score_ub)

def confusion_matrix(model, X, y, labels=None):
    '''
    Compute the confusion matrix for a classification model using
    test data (X,y)
    Parameters:
      model -- a classification model
      X     -- sklearn style feature matrix
      y     -- sklearn style target vector
      labels -- list of class labels
    Returns a Pandas dataframe holding the confusion matrix
    '''
    if not labels:
        labels = list(model.classes_)
    if len(labels) != y.value_counts().shape[0]:
        raise ValueError('labels must match the number of classes in y')
    y_pred = model.predict(X)
    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)  

#################################################################################
# clustering

def plot_elbow(X, n=10):
    """
    Generates an elbow plot for KMeans clustering.
    
    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame)
    - n: Max number of clusters to consider (int)

    The function fits KMeans for 1 to k clusters and plots the average within-cluster variance.
    """
    plt.show() # flush any previous plots

    # Loop over number of clusters from 1 to k
    average_variances = []
    for k in range(1, n + 1):
        kmeans = sklearn.cluster.KMeans(k).fit(X)
        
        # Compute average within-cluster variance
        #avg_variance = kmeans.inertia_ / X.shape[0]
        avg_variance = kmeans.inertia_ / k
        average_variances.append(avg_variance)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k + 1), average_variances, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Within-Cluster Variance')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(range(1, k + 1))
    plt.grid(True)
    plt.show()


#################################################################################
# NLP

def docterm_matrix(docs, 
                   doc_names=None, 
                   min_df=1,
                   token_pattern='(?u)\\b\\w+\\b',
                   stem=False,
                   stop_words=None):
    '''
    Compute a document-term matrix from a list of documents.
    docs - a list of documents
    doc_names - a list of document names
    min_df - minimum document frequency
    token_pattern - token pattern
    stem - if True stem the terms
    stop_words - None or a list of stop words or 'english'
    '''
    if not doc_names:
      doc_names = [f'doc{i}' for i in range(len(docs))]

    doc_analyzer = CountVectorizer(analyzer = "word",
                              stop_words = stop_words,
                              token_pattern = token_pattern) \
          .build_analyzer() # retrieve the analyzer
    if stem:
      stemmer = PorterStemmer()
      analyzer = lambda doc: [stemmer.stem(w) for w in doc_analyzer(doc)]
    else:
      analyzer = doc_analyzer
      
    vectorizer = CountVectorizer(analyzer=analyzer, # use our doc stemmer function
                                 binary=True,
                                 min_df=min_df)
    doc_array = vectorizer.fit_transform(docs).toarray()
    doc_features = vectorizer.get_feature_names_out()
    docterm = pd.DataFrame(data=doc_array,
                           index=doc_names,
                           columns=doc_features)
    return docterm

#################################################################################
# database utilities

class DBCredentials:
  '''
  Allows the user to instantiate DB credential objects for the
  use in the 'execute_query' function.

  The defaults are set to point to the default DB for the CSC310
  course at URI.
  '''
  def __init__(self,
               host = 'testdb.cwy05wfzuxbv.us-east-1.rds.amazonaws.com',
               userdb = 'world',
               user = 'csc310',
               password = 'csc310$is$fun'):
    self.host = host
    self.userdb = userdb
    self.user = user
    self.password = password

def execute_query(credentials, sql_string):
  '''
  execute an sql query
    credentials - a instantiated DBCredentials object
    sql_string  - a string holding the SQL query
  
  Returns a Pandas dataframe holding the result table
  '''
  db = sql.connect(host=credentials.host,
                   user=credentials.user,
                   password=credentials.password,
                   database=credentials.userdb)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = pd.read_sql(sql_string, con=db)
  db.close()
  return data

#################################################################################
# testing
if __name__ == '__main__':
  # classification
  df = pd.read_csv("abalone.csv")
  X = df.drop(columns=['sex'])
  y = df[['sex']]
  tree = sklearn.tree.DecisionTreeClassifier(max_depth=3).fit(X,y)
  print("bootstrap confint: {}".format(bootstrap_score(tree,X,y,as_string=True)))
  print("estimated confint: {}".format(acc_score(tree,X,y,as_string=True)))
  print(confusion_matrix(tree,X,y))

  # elbow plot
  df = pd.read_csv("iris.csv")
  X = df.drop(columns=['Species'])
  plot_elbow(X)

  # NLP
  docs = ["the quick brown fox jumps over the lazy dog",
        "rudi is a lazy brown dog",
        "princess jumps over the lazy dog"]
  docterm = docterm_matrix(docs)
  print(docterm)

