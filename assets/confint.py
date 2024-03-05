# compute 95% confidence intervals for classification and regression
# problems

def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    if acc > 1.0:
      raise ValueError('Expected an accuracy value between 0 and 1')
    import math
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
    import math
    interval = 2*math.sqrt((4*rs_score*(1-rs_score)**2*(n-k-1)**2)/((n**2 - 1)*(n+3)))
    lb = max(0, rs_score - interval)
    ub = min(1.0, rs_score + interval)
    return (lb,ub)
