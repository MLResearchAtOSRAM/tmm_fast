import numpy as np

def get_n_from_txt(filepath, points=None, lambda_min=400, lambda_max=700, complex_n=True):
    ntxt = np.loadtxt(filepath)
    if np.min(np.abs(ntxt[:, 0] - lambda_min)) > 25 or np.min(np.abs(ntxt[:, 0] - lambda_max)) > 25:
        print('No measurement data for refractive indicies are available within 25 nm in \n' + filepath)
    if points is None:
        points = lambda_max - lambda_min + 1
    idxmin = np.argmin(np.abs(ntxt[:, 0] - lambda_min))
    idxmax = np.argmin(np.abs(ntxt[:, 0] - lambda_max))
    if idxmax == idxmin:
        if complex_n:
            indicies = np.vectorize(complex)(np.array([ntxt[idxmin, 1]]), np.array([ntxt[idxmin, 2]]))
        else:
            indicies = np.array([ntxt[idxmin, 1]])
    else:
        xp = ntxt[idxmin:idxmax, 0]
        fpn = ntxt[idxmin:idxmax, 1]
        n = np.interp(np.linspace(lambda_min, lambda_max, points), xp, fpn)
        if complex_n:
            fpk = ntxt[idxmin:idxmax, 2].squeeze()
            k = np.interp(np.linspace(lambda_min, lambda_max, points), xp, fpk)
            indicies = np.vectorize(complex)(n, k)
        else:
            indicies = n
    return indicies

def get_N(path_list, lambda_min, lambda_max, points=None, complex_n=False):
    n = []
    for path in path_list:
        n.append(get_n_from_txt(path, points, lambda_min=lambda_min, lambda_max=lambda_max, complex_n=complex_n))
    return np.vstack((n))
