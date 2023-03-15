from simba.unsupervised.misc import (read_pickle,
                                     find_embedding,
                                     check_directory_exists)
from simba.misc_tools import SimbaTimer
import numpy as np
from numba import jit, prange
from numba.typed import List
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph


class DBCVCalculator(object):
    """
    Density-based Cluster Validation (DBCV).

    Parameters
    ----------
    embedders_path: str
        Directory holding dimensionality reduction models in pickle format.
    clusterers_path: str
        Directory holding cluster models in pickle format.

    Notes
    -----
    Numbafied version of `DBCSV GitHub repository <https://github.com/christopherjenness/DBCV>`__.
    In part increases speed by replacing `scipy.spatial.distance.cdist` in original DBCSV with LLVM as discussed here
    `<https://github.com/numba/numba-scipy/issues/38>`__.

    References
    ----------

    .. [1] Moulavi et al, Density-Based Clustering Validation, `SIAM 2014`, https://doi.org/10.1137/1.9781611973440.96

    Examples
    -----
    >>> dbcv_calculator = DBCVCalculator(embedders_path='unsupervised/dr_models', clusterers_path='unsupervised/cluster_models')
    >>> results = dbcv_calculator.run()
    """

    def __init__(self,
                 embedders_path: str,
                 clusterers_path: str):

        check_directory_exists(embedders_path)
        check_directory_exists(clusterers_path)
        print('Reading in data...')
        self.embedders = read_pickle(data_path=embedders_path)
        self.clusterers = read_pickle(data_path=clusterers_path)
        print(f'Analyzing DBCV for {len(self.clusterers.keys())} clusterers...')
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def run(self):
        results = {}
        for k, v in self.clusterers.items():
            print(f"Performing DBCV for model {v['NAME']}...")
            model_timer = SimbaTimer()
            model_timer.start_timer()
            results[k] = {}
            embedder = find_embedding(embeddings=self.embedders, hash=v['HASH'])
            labels = v['model'].labels_.reshape(-1, 1).astype(np.int8)
            unique_labels = np.unique(labels).shape[0]
            X = embedder['models'].embedding_
            results[k]['clusterer_name'] = v['NAME']
            results[k]['embedder_name'] = embedder['HASH']
            results[k] = {**results[k], **v['parameters']}
            dbcv = 'nan'
            if unique_labels > 1:
                dbcv = self.DBCV(X, labels)
            results[k] = {**results[k], **{'dbcv': dbcv}}
            results[k] = {**results[k], **{'cluster_cnt': unique_labels}}
            model_timer.stop_timer()
            print(f"DBCV complete for model {results[k]['clusterer_name']} (elapsed time {model_timer.elapsed_time_str}s)...")
        return results

    def DBCV(self, X, labels):
        """
        Parameters
        ----------
        X : array with shape len(observations) x len(dimentionality reduced dimensions)
        labels : 1D array with cluster labels

        Returns
        -------
        DBCV_validity_index: DBCV cluster validity score

        """

        neighbours_w_labels, ordered_labels = self._mutual_reach_dist_graph(X, labels)
        graph = self.calculate_dists(X, neighbours_w_labels)
        mst = self._mutual_reach_dist_MST(graph)
        return self._clustering_validity_index(mst, ordered_labels)

    @staticmethod
    @jit(nopython=True)
    def _mutual_reach_dist_graph(X, labels):
        """
        Parameters
        ----------
        X :  array with shape len(observations) x len(dimensionality reduced dimensions)
        labels : 1D array with cluster labels

        Returns
        -------
        arrays_by_cluster: 3D array in numba ListType, of shape len(number of clusters) x len(cluster_members) x len(dimensionality reduced dimensions)
        ordered_labels:  numba ListType of labels in the order of arrays_by_cluster
        """
        unique_cluster_labels = np.unique(labels)
        arrays_by_cluster = List()
        ordered_labels = List()

        def _get_label_members(X, labels, cluster_label):
            indices = np.where(labels == cluster_label)[0]
            members = X[indices]
            return members

        for cluster_label_idx in prange(unique_cluster_labels.shape[0]):
            cluster_label = unique_cluster_labels[cluster_label_idx]
            members = _get_label_members(X, labels, cluster_label)
            ordered_labels.append([cluster_label] * len(members))
            arrays_by_cluster.append([members])

        ordered_labels = [i for s in ordered_labels for i in s]
        return arrays_by_cluster, ordered_labels

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def calculate_dists(X, arrays_by_cluster):
        """
        Parameters
        ----------
        arrays_by_cluster: 3D array in numba ListType, of shape len(number of clusters) x len(cluster_members) x len(dimentionality reduced dimensions

        Returns
        -------
        graph: Graph of all pair-wise mutual reachability distances between points.

        """

        cluster_n = len(arrays_by_cluster)
        graph = np.empty((X.shape[0], X.shape[0]), X.dtype)
        graph_row_counter = 0

        for cluster_a in range(cluster_n):
            A = arrays_by_cluster[cluster_a][0]
            for a in prange(A.shape[0]):
                Ci = np.empty((A.shape[0]), A.dtype)
                init_val_arr = np.zeros(1, A.dtype)
                init_val = init_val_arr[0]
                for j in range(A.shape[0]):
                    acc = init_val
                    for k in range(A.shape[1]):
                        acc += (A[a, k] - A[j, k]) ** 2
                    Ci[j] = np.sqrt(acc)

                indices = np.where(Ci != 0)
                Ci = Ci[indices]

                numerator = ((1 / Ci) ** A.shape[1]).sum()
                core_dist_i = (numerator / (A.shape[0] - 1)) ** (-1 / A.shape[1])
                graph_row = np.zeros((0))

                for array_cnt in range(len(arrays_by_cluster)):
                    B = arrays_by_cluster[array_cnt][0]
                    for b in prange(B.shape[0]):
                        Cii = np.empty((B.shape[0]), B.dtype)
                        init_val_arr = np.zeros(1, B.dtype)
                        init_val = init_val_arr[0]
                        for j in range(B.shape[0]):
                            acc = init_val
                            for k in range(B.shape[1]):
                                acc += (B[b, k] - B[j, k]) ** 2
                            Cii[j] = np.sqrt(acc)
                        indices_ii = np.where(Cii != 0)
                        Cii = Cii[indices_ii]

                        numerator = ((1 / Cii) ** B.shape[1]).sum()
                        core_dist_j = (numerator / (B.shape[0] - 1)) ** (-1 / B.shape[1])
                        dist = np.linalg.norm(A[a] - B[b])
                        mutual_reachability = np.max(np.array([core_dist_i, core_dist_j, dist]))
                        graph_row = np.append(graph_row, mutual_reachability)

                graph[graph_row_counter] = graph_row
                graph_row_counter += 1

        return graph

    def _mutual_reach_dist_MST(self, dist_tree):
        """
        Parameters
        dist_tree : array of dimensions len(observations) x len(observations


        Returns
        -------
        minimum_spanning_tree: array of dimensions len(observations) x len(observations, minimum spanning tree of all pair-wise mutual reachability distances between points.

        """

        mst = minimum_spanning_tree(dist_tree).toarray()
        return self.transpose_np(mst)

    @staticmethod
    @jit(nopython=True)
    def transpose_np(mst):
        return mst + np.transpose(mst)


    def _clustering_validity_index(self, MST, labels):

        """
        Parameters
        MST: minimum spanning tree of all pair-wisemutual reachability distances between points
        labels : 1D array with cluster labels

        Returns
        -------
        validity_index : float score in range[-1, 1] indicating validity of clustering assignments

        """
        n_samples = len(labels)
        validity_index = 0

        for label in np.unique(labels):
            fraction = np.sum(labels == label) / float(n_samples)
            cluster_validity = self._cluster_validity_index(MST, labels, label)
            validity_index += fraction * cluster_validity

        return validity_index

    def _cluster_validity_index(self, MST, labels, cluster):
        min_density_separation = np.inf
        for cluster_j in np.unique(labels):
            if cluster_j != cluster:
                cluster_density_separation = self._cluster_density_separation(MST, labels, cluster, cluster_j)
                if cluster_density_separation < min_density_separation:
                    min_density_separation = cluster_density_separation

        cluster_density_sparseness = self._cluster_density_sparseness(MST, np.array(labels), np.array(cluster))
        numerator = min_density_separation - cluster_density_sparseness
        denominator = np.max([min_density_separation, cluster_density_sparseness])
        cluster_validity = numerator / denominator
        return cluster_validity

    def _cluster_density_separation(self, MST, labels, cluster_i, cluster_j):
        indices_i = np.where(labels == cluster_i)[0]
        indices_j = np.where(labels == cluster_j)[0]
        shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
        relevant_paths = shortest_paths[:, indices_j]
        density_separation = np.min(relevant_paths)
        return density_separation

    @staticmethod
    @jit(nopython=True)
    def _cluster_density_sparseness(MST, labels, cluster):
        indices = np.where(labels == cluster)[0]
        cluster_MST = MST[indices][:, indices]
        cluster_density_sparseness = np.max(cluster_MST)
        return cluster_density_sparseness

test = DBCVCalculator(embedders_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models',
                      clusterers_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models')
dbcv = test.run()
