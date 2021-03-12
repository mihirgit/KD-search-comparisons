# Mihir G
# Ayush K

import multiprocessing as processing
import time
import numpy
import ctypes
import os


def kdtree(data, leafsize=10):
    """
    build a kd-tree for O(n log n) nearest neighbour search

    input:
        data:       2D ndarray, shape =(ndim,ndata), preferentially C order
        leafsize:   max. number of data points to leave in a leaf

    output:
        kd-tree:    list of tuples
    """

    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = numpy.zeros((2, data.shape[0]))
    hrect[0, :] = data.min(axis=1)
    hrect[1, :] = data.max(axis=1)

    # create root of kd-tree
    idx = numpy.argsort(data[0, :], kind='mergesort')
    data[:, :] = data[:, idx]
    splitval = data[0, ndata // 2]  # // added for integer

    left_hrect = hrect.copy()
    right_hrect = hrect.copy()
    left_hrect[1, 0] = splitval
    right_hrect[0, 0] = splitval

    tree = [(None, None, left_hrect, right_hrect, None, None)]

    stack = [(data[:, :ndata // 2], idx[:ndata // 2], 1, 0, True),
             (data[:, ndata // 2:], idx[ndata // 2:], 1, 0, False)]

    # recursively split data in halves using hyper-rectangles:
    while stack:

        # pop data off stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        nodeptr = len(tree)

        # update parent node

        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

        tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
            else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

        # insert node in kd-tree

        # leaf node?
        if ndata <= leafsize:
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            tree.append(leaf)

        # not a leaf, split the data in two
        else:
            splitdim = depth % ndim
            idx = numpy.argsort(data[splitdim, :], kind='mergesort')
            data[:, :] = data[:, idx]
            didx = didx[idx]
            nodeptr = len(tree)
            stack.append((data[:, :ndata // 2], didx[:ndata // 2], depth + 1, nodeptr, True))
            stack.append((data[:, ndata // 2:], didx[ndata // 2:], depth + 1, nodeptr, False))
            splitval = data[splitdim, ndata // 2]
            if leftbranch:
                left_hrect = _left_hrect.copy()
                right_hrect = _left_hrect.copy()
            else:
                left_hrect = _right_hrect.copy()
                right_hrect = _right_hrect.copy()
            left_hrect[1, splitdim] = splitval
            right_hrect[0, splitdim] = splitval
            # append node to tree
            tree.append((None, None, left_hrect, right_hrect, None, None))

    return tree


def intersect(hrect, r2, centroid):
    """
    checks if the hyperrectangle hrect intersects with the
    hypersphere defined by centroid and r2
    """
    maxval = hrect[1, :]
    minval = hrect[0, :]
    p = centroid.copy()
    idx = p < minval
    p[idx] = minval[idx]
    idx = p > maxval
    p[idx] = maxval[idx]
    return ((p - centroid) ** 2).sum() < r2


def quadratic_knn_search(data, lidx, ldata, K):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    sqd = ((ldata - data[:, :ndata]) ** 2).sum(axis=0)  # data.reshape((param,1)).repeat(ndata, axis=1);
    # sqd = ((ldata - data[:, :ndata]) ** 2).sum(axis=0) original code for above line
    idx = numpy.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])


def search_kdtree(tree, datapoint, K):
    """ find the k nearest neighbours of datapoint in a kdtree """
    stack = [tree[0]]
    knn = [(numpy.inf, None)] * K
    _datapt = datapoint[:, 0]
    while stack:

        leaf_idx, leaf_data, left_hrect, \
        right_hrect, left, right = stack.pop()

        # leaf
        if leaf_idx is not None:
            _knn = list(quadratic_knn_search(datapoint, leaf_idx, leaf_data, K))
            if _knn[0][0] < knn[-1][0]:
                knn = sorted(knn + _knn)[:K]

        # not a leaf
        else:

            # check left branch
            if intersect(left_hrect, knn[-1][0], _datapt):
                stack.append(tree[left])

            # chech right branch
            if intersect(right_hrect, knn[-1][0], _datapt):
                stack.append(tree[right])
    return knn


def knn_search(data, K, leafsize=2048):
    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=leafsize)

    # search kdtree
    knn = []
    for i in numpy.arange(ndata):
        _data = data[:, i].reshape((param, 1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K + 1)
        knn.append(_knn[1:])

    return knn


# Parallel Search

def __num_processors():
    if os.name == 'nt':  # Windows
        return int(os.getenv('NUMBER_OF_PROCESSORS'))
    else:  # glibc (Linux, *BSD, Apple)
        get_nprocs = ctypes.cdll.libc.get_nprocs
        get_nprocs.restype = ctypes.c_int
        get_nprocs.argtypes = []
        return get_nprocs()


def __search_kdtree(tree, data, K, leafsize):
    knn = []
    param = data.shape[0]
    ndata = data.shape[1]
    for i in numpy.arange(ndata):
        _data = data[:, i].reshape((param, 1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K + 1)
        knn.append(_knn[1:])
    return knn


def __remote_process(rank, qin, qout, tree, K, leafsize):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        # process data
        knn = __search_kdtree(tree, data, K, leafsize)
        # write to output queue
        qout.put((nc, knn))


def knn_search_parallel(data, K, leafsize=2048):
    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree, exploiting all logical
        processors on the computer """

    ndata = data.shape[1]
    param = data.shape[0]
    nproc = __num_processors()
    # build kdtree
    tree = kdtree(data.copy(), leafsize=leafsize)
    # compute chunk size
    chunk_size = data.shape[1] // (4 * nproc)
    chunk_size = 100 if chunk_size < 100 else chunk_size
    # set up a pool of processes
    qin = processing.Queue(maxsize=ndata // chunk_size)
    qout = processing.Queue(maxsize=ndata // chunk_size)
    pool = [processing.Process(target=__remote_process,
                               args=(rank, qin, qout, tree, K, leafsize))
            for rank in range(nproc)]
    for p in pool:
        p.start()
    # put data chunks in input queue
    cur, nc = 0, 0
    while 1:
        _data = data[:, cur:cur + chunk_size]
        if _data.shape[1] == 0:
            break
        qin.put((nc, _data))
        cur += chunk_size
        nc += 1
    # read output queue
    knn = []
    while len(knn) < nc:
        knn += [qout.get()]
    # avoid race condition
    _knn = [n for i, n in sorted(knn)]
    knn = []
    for tmp in _knn:
        knn += tmp
    # terminate workers
    for p in pool:
        p.terminate()
    return knn


if __name__ == '__main__':
    K = 15  # original = 11     (k = no. of nearest neighbours to find in tree)
    Ndata = 25000  # original = 10000  ( data points )
    Ndim = 6  # original = 12        ( dimension of kd tree ) (range 3 to 20)
    data = 10 * numpy.random.rand(Ndata * Ndim).reshape((Ndim, Ndata))
    # t0 = time.time_ns()
    # knn_search_parallel(data, K)
    # t1 = time.time_ns()
    t2 = time.time_ns()
    knn_search(data, K)
    t3 = time.time_ns()
    # t4 = time.time()
    # brute_force_quadratic_search(data, K)
    # t5 = time.time_ns()
    # parallel_time = t1 - t0
    knn_time = t3 - t2
    # brute_force_time = t5 - t4
    # print("Parallel Search Time: ", parallel_time)
    print("Knn Search time: ", knn_time)
    # print("Brute Force Search Time: ", brute_force_time)
    # t6 = time.time_ns()
    # kdtree(data)
    # t7 = time.time_ns()
    # print("Time to construct tree: ", t7-t6)
