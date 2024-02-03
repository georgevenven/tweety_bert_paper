
def generate_optics_labels(array, min_samples=5, xi=0.05, min_cluster_size=0.1):
    """
    Generate labels for data points using the OPTICS (Ordering Points To Identify the Clustering Structure) clustering algorithm, 
    with adjusted parameters to reduce the number of points marked as noise.

    Parameters:
    - array: ndarray of shape (n_samples, n_features)
      The input data to cluster.

    - min_samples: int, default=10
      The number of samples in a neighborhood for a point to be considered as a core point.

    - xi: float, between 0 and 1, default=0.05
      Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.

    - min_cluster_size: float or int, default=0.1
      The minimum size of a cluster. If it is less than 1, it is interpreted as a fraction of the total number of points.

    Returns:
    - labels: ndarray of shape (n_samples)
      Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    """

    from sklearn.cluster import OPTICS

    # Create an OPTICS object with the specified parameters.
    optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)

    # Fit the model to the data and extract the labels.
    optics_model.fit(array)
    labels = optics_model.labels_

    return labels

def generate_kmeans_labels(array, n_clusters=10):
    """
    Applies K-means clustering to an array of data points and returns the cluster labels.

    Args:
        array (numpy.ndarray): An n x features array of data points.
        n_clusters (int): The number of clusters to form.

    Returns:
        numpy.ndarray: An n x 1 array of cluster labels for each data point.
    """

    # Initialize the KMeans clusterer
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the model to the data and predict the cluster labels
    labels = kmeans.fit_predict(array)

    # Optionally, reshape labels to n x 1 if you specifically need this shape
    labels = labels.reshape(-1, 1)

    # Print the unique labels to see the different clusters
    print(np.unique(labels))

    return labels

def generate_hmm_states(array, n_states=30, context=1000):
    """
    Applies Hidden Markov Model to an array of data points and returns the state labels.

    Args:
        array (numpy.ndarray): An n x features array of data points, representing a sequence of observations.
        n_states (int): The number of hidden states in the model.

    Returns:
        numpy.ndarray: An n x 1 array of state labels for each observation.
    """

    # Initialize the HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",  init_params='')

    # Initialize starting probabilities to be equal
    model.startprob_ = np.array([1.0 / n_states] * n_states)

    # Initialize the transition matrix with a bias for staying in the same state
    transition_matrix = np.eye(n_states) * 0.9
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                transition_matrix[i, j] = 0.1 / (n_states - 1)

    model.transmat_ = transition_matrix

    # Use K-means clustering to initialize the means
    kmeans = KMeans(n_clusters=n_states).fit(array)
    model.means_ = kmeans.cluster_centers_

    # Initialize the covariance matrices based on cluster assignments
    covars = np.zeros((n_states, array.shape[1], array.shape[1]))
    for i in range(n_states):
        cluster_points = array[kmeans.labels_ == i]
        if len(cluster_points) > 1:  # Ensure there are enough points to calculate covariance
            covars[i] = np.cov(cluster_points, rowvar=False)
        else:  # Fallback to identity if cluster has insufficient points
            covars[i] = np.identity(array.shape[1])
    model.covars_ = covars

    # Calculate lengths of sequences if the dataset contains multiple sequences
    lengths = [context] * (len(array) // context)

    # Fit the model to the data
    model.fit(array, lengths=lengths)

    # Predict the hidden states for each observation in the array
    states = model.predict(array)

    # Optionally, reshape states to n x 1 if you specifically need this shape
    states = states.reshape(-1, 1)

    # Print the unique states to see the different inferred hidden states
    print(np.unique(states))

    return states