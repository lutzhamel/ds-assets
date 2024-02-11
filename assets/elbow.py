# inspired by the source code from
# www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn
def plot_elbow(X, n=10):
    import numpy as np
    import seaborn as sns
    from matplotlib.pyplot import show, xlabel, ylabel
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    sns.set()
    show() # show anything that is already on the display stack.
    
    # kmeans models for each k
    kMeansModels = [KMeans(n_clusters=k, n_init='auto').fit(X) for k in range(1, n+1)]
    
    # coordinates of the centroids of the models
    centroids = [m.cluster_centers_ for m in kMeansModels]
    
    # find the distances of the values to the centroids
    k_euclid = [cdist(X, cent) for cent in centroids]
    
    # find the distance of each point to its cluster center
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    
    # average variance for each cluster configuration
    dist_tuple = zip(list(range(1,n+1)),dist)
    average_var = [sum(d**2)/k for (k,d) in dist_tuple]

    # plot the variance of the models
    sns.lineplot(x=list(range(1,n+1)), y=average_var)
    xlabel('k')
    ylabel('Average Cluster Variance')
    show()
