import pandas as pd
import statistics as stats
from numpy import percentile
from sklearn.model_selection import train_test_split
import warnings


def bootstrap(model, X, y, random_state=None):
    '''
    Compute a bootstrapped model score together with its 95% probability 
    bound. If the model object is a classification model then model accuracy 
    is computed and if the model object is a regression model then the R^2 
    score is computed.

    Parameters
        model - either a classification or regression model
        X - sklearn style feature matrix
        y - sklearn style target vector
        random_state - controls the sampling of the bootstrap samples, 
                       pass an int for reproducible output across multiple 
                       function calls.
    
    Note: if no validation data is given then the training data is 
    used for testing.

    Returns
        (score, lower bound, upper bound) 
    '''
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    rows = X.shape[0]
    D = pd.concat([X,y], axis=1)
    score_list = []
    for i in range(200):
        if not random_state:
            B = D.sample(n=rows,replace=True)
        else:
            B = D.sample(n=rows,replace=True,random_state=random_state+i)
        BX = B.drop(columns=y.columns)
        By = B[y.columns]
        warnings.filterwarnings('ignore')
        train_test_split(BX,By,train_size=0.7,test_size=0.3,random_state=random_state)
        model.fit(BX, By)
        score_list.append(model.score(BX, By))
        warnings.filterwarnings('always')
    score_list.sort()
    score_avg = stats.mean(score_list)
    score_ub = percentile(score_list,97.5)
    score_lb = percentile(score_list,2.5)
    return (score_avg, score_lb, score_ub)

if __name__ == '__main__':
    from sklearn import tree

    # classification
    t1c = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)

    print("******** abalone ***********")
    df = pd.read_csv("abalone.csv")
    X = df.drop(columns=['sex'])
    y = df[['sex']]
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1c,X,y)))
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1c,X,y,random_state=1)))

    # regression
    t1r = tree.DecisionTreeRegressor(max_depth=3)

    print("******** cars ***********")
    df = pd.read_csv("cars.csv")
    X = df.drop(columns=['dist'])
    y = df['dist']
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1r,X,y)))
    print("Confidence interval max_depth=3: {}".format(bootstrap(t1r,X,y,random_state=1)))

