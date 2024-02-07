# inspired by the source code from
# www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn
def plot_elbow(X, n=10):
    import seaborn as sns
    sns.set()
    from matplotlib.pyplot import show, xlabel, ylabel
    show() # show anything that is already on the display stack.
    from sklearn.cluster import KMeans
    import numpy as np
    from scipy.spatial.distance import cdist, pdist
    
    # kmeans models for each k
    kMeansModels = [KMeans(n_clusters=k, n_init='auto').fit(X) for k in range(1, n+1)]
    
    # get the centroids of the models
    centroids = [m.cluster_centers_ for m in kMeansModels]
    
    # find the distances of the values to the centroids
    k_euclid = [cdist(X, cent) for cent in centroids]
    
    # find the distance of each point to its cluster center
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    
    # total within cluster sum of squares
    wcss = [sum(d**2) for d in dist]

    # plot the variance of the models
    #plt.plot(list(range(1,n+1)),wcss)
    sns.lineplot(x=list(range(1,n+1)), y=wcss)
    xlabel('k')
    ylabel('Within Cluster Variance')
    show()
