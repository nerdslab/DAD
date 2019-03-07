import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import sqrtm
import time
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from decoder import utils


def run_DAD_3D(X_train, Y_test, T_train, X_test, T_test, grid_size=5, dim_red_method="PCA", num_T=0, check_2D=False):
    k = 3  # align in 3 dimensions

    if (k == 3) and (np.size(X_train, 1) == 2):
        X_train = map_X_3D(X_train)

    X_n = normal(X_train)
    Y_r = remove_const_cols(Y_test)

    t1 = time.time()
    V_r = compute_V(Y_r, map_X_3D(X_test), T_test, d=k, methods=[dim_red_method])
    t2 = time.time()

    print(f'Finished computing the low-d embedding in {(t2 - t1) / 60:.2f} minutes.')

    X_rec = []
    V_out = []
    V_flip = []
    min_KL = []

    for i in range(len(V_r)):
        X_rec_tmp, V_out_tmp, V_flip_tmp, min_KL_tmp = DAD_3D_search(X_n, normal(V_r[i]), grid_size, num_T, check_2D)
        X_rec.append(X_rec_tmp)
        V_out.append(V_out_tmp)
        V_flip.append(V_flip_tmp)
        min_KL.append(min_KL_tmp)

    return np.array(X_rec)[0, :, :]


def map_X_3D(X):
    return np.column_stack((X[:, 0], X[:, 1], np.linalg.norm(X, axis=1)))


def normal(X):
    mean_X = np.mean(X, axis=0)
    cov_X = np.cov(X, rowvar=False)
    X_n = X
    #for col in range(X.shape[1]):
    #    X_n[:, col] = (X_n[:, col] - mean_X[col]) / np.sqrt(cov_X[col, col])
    return np.matmul(X_n - mean_X, np.linalg.inv(sqrtm(cov_X)))


def remove_const_cols(Y):
    return Y[:, ~np.all(Y[1:] == Y[:-1], axis=0)]


def compute_V(Y, X_test, T_test, d=3, methods=['PCA', 'MDS', 'FA', 'Isomap']):
    L = len(methods)
    V = []
    for idx in range(L):
        if methods[idx] == 'PCA':
            pca = PCA(n_components=d)
            V.append(pca.fit_transform(Y))
        elif methods[idx] == 'MDS':
            mds = MDS(n_components=d)
            V.append(mds.fit_transform(Y))
        elif methods[idx] == 'FA':
            fa = FactorAnalysis(n_components=d)
            V.append(fa.fit_transform(Y))
        elif methods[idx] == 'Isomap':
            isomap = Isomap(n_components=d)
            V.append(isomap.fit_transform(Y))

    plt.figure(figsize=(8,6))
    plt.subplot(1, 2, 1)
    utils.color_data(X_test, T_test)
    plt.title('Ground Truth')
    plt.subplot(1, 2, 2)
    utils.color_data(V[0], T_test)
    plt.title('FA')
    plt.show()

    return V


def DAD_3D_search(X_n, V_r, grid_size=8, num_T=1, check_2D=False):
    t1 = time.time()
    V_out = grid_search_3D_KL(X_n, V_r, grid_size, num_T)
    t2 = time.time()

    print(f'Finished performing 3D alignment in {(t2 - t1) / 60:.2f} minutes.')

    num_ang = 90

    t1 = time.time()
    X_rec_3D, V_flip_3D, y_3D, inds_3D = rotated_KL_min(V_out[:, 0:2], X_n[0:2, :], num_ang)
    t2 = time.time()

    print(f'Finished performing the final 2D rotation in {(t2 - t1) / 60:.2f} minutes.')

    if check_2D:
        X_rec_2D, V_flip_2D, y_2D, inds_2D = rotated_KL_min(V_r[:, 0:2], X_n[:, 0:2], num_ang)

        if np.amin(y_3D[inds_3D]) < np.amin(y_2D[inds_2D]):
            X_rec = X_rec_3D
            min_KL = y_3D[inds_3D]
        else:
            X_rec = X_rec_2D
            min_KL = y_2D[inds_2D]
            V_out = V_r

        V_flip = []
        sort_inds = np.argsort(np.hstack((y_2D[inds_2D], y_3D[inds_3D])))

        for i in range(10):
            if sort_inds[i] < inds_2D.size:
                V_flip[i] = V_flip_2D[sort_inds[i]]
            else:
                V_flip[i] = V_flip_3D[sort_inds[i]]

    else:
        V_flip = V_flip_3D
        X_rec = X_rec_3D
        min_KL = y_3D[inds_3D]

    return X_rec, V_out, V_flip, min_KL


def grid_search_3D_KL(X_target, Y_source, num_A, num_T):
    mean_weight = 0.7
    KL_thr = 5
    nz_var = 0.5
    fine_grid = 10
    bsz = 50
    num_samples = 500000
    k0 = k1 = 5
    grid_size = num_A

    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))

    F_mat = np.column_stack((xx.flatten('F'), yy.flatten('F'), zz.flatten('F')))
    F_mat = np.concatenate((np.array([0, 0, 1])[np.newaxis, :], F_mat[np.linalg.norm(F_mat, ord=2, axis=1) > 0.1, :]), axis=0)

    if num_T > 1:
        t_vec = np.vstack(([0, 0, 0], np.random.randn(num_T, 3) * nz_var))
    else:
        t_vec = np.array([0, 0, 0])

    sample_loc = sample_from_3D_grid(bsz, num_samples)
    p_train = prob1(sample_loc, normal(X_target), k0)

    try:
        dists1 = np.load('dists1.npy')
    except IOError:
        if num_T > 0:
            dists1 = np.full((F_mat.shape[0], num_T if num_T > 1 else 1), 0.0)
        else:
            dists1 = np.full(F_mat.shape[0], 0.0)

        for i in range(F_mat.shape[0]):
            an0 = F_mat[i, :]
            Y_rot = rotate_data(Y_source, an0)

            p_rot = prob1(sample_loc, normal(Y_rot), k1)

            if num_T > 0:
                dists1[i, 0] = np.matmul(p_rot.T, np.log(p_rot / p_train))
                if dists1[i, 0] < KL_thr and num_T > 1:
                    for j in range(1, num_T):
                        Y_rot2 = Y_rot + t_vec[j, :]
                        p_rot = prob1(sample_loc, Y_rot2, k1)
                        dists1[i, j] = np.matmul(p_rot.T, np.log(p_rot / p_train))

                    KL_thr = min(KL_thr, np.mean(dists1[dists1 != 100]) * mean_weight)
            else:
                dists1[i] = np.matmul(p_rot, np.log(p_rot / p_train))

    np.save('dists1.npy', dists1)
    plt.plot(dists1)
    plt.title('3D Grid Search')
    plt.xlabel('Rotation Angle')
    plt.ylabel('KL Divergence')
    plt.show()
    # select best angle of rotation
    values = np.amin(dists1, axis=0)
    ind = np.argmin(dists1, axis=0)

    if num_T > 1:
        ind = ind[np.argmin(values)]

    angle_ind = ind

    an0 = F_mat[angle_ind, :]
    Y_curr = rotate_data(Y_source, an0)

    if num_T > 1:
        t_curr = t_vec[ind, :]
    else:
        t_curr = [0, 0, 0]

    # final translation
    t_vec2 = np.random.randn(np.power(fine_grid, 3), 3) * nz_var + np.matlib.repmat(t_curr, np.power(fine_grid, 3), 1)

    dists2 = np.zeros(t_vec2.shape[0])
    for i in range(t_vec2.shape[0]):
        Y_rot2 = Y_curr + np.matlib.repmat(t_vec2[i, :], Y_curr.shape[0], 1)
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_target)
        distances, dvec = nbrs.kneighbors(Y_rot2)
        dists2[i] = np.mean(dvec)

    ind = np.argmin(dists2)
    return Y_curr + np.matlib.repmat(t_vec2[ind,:], Y_curr.shape[0], 1)


def sample_from_3D_grid(bsz, num_samples, x_min=-4, x_max=4, y_min=-4, y_max=4, z_min=-4, z_max=4):
    x1, y1, z1 = np.meshgrid(np.linspace(x_min, x_max, bsz - 1), np.linspace(y_min, y_max, bsz - 1), np.linspace(z_min, z_max, bsz - 1))
    x_t = np.column_stack((x1.flatten('F'), y1.flatten('F'), z1.flatten('F')))
    N = x_t.shape[0]

    return x_t[np.random.permutation(N)[0:min(num_samples, N)], :]


def prob1(X1, x_t, k=1):
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_t)
    distances, indices = nbrs.kneighbors(X1)
    rho_X1 = distances[:, -1]
    p1 = k / (X1.shape[0] * np.power(rho_X1, np.full(rho_X1.shape[0], x_t.shape[1])))
    return p1 / p1.sum()


def rotate_data(Y_curr, an0):
    an0 = an0 / np.linalg.norm(an0)
    v1 = [0, 0, 1]
    if np.array_equal(v1, an0):
        rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif np.array_equal(v1, -an0):
        rot_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        rot_mat = np.array([[np.dot(v1, an0), -np.linalg.norm(np.cross(v1, an0)), 0], [np.linalg.norm(np.cross(v1, an0)), np.dot(v1, an0), 0], [0, 0, 1]])
        basis_mat = np.column_stack((v1, (an0 - v1) / np.linalg.norm(an0 - v1), np.cross(an0, v1)))
        try:
            rot_mat = np.matmul(np.matmul(basis_mat, rot_mat), np.linalg.inv(basis_mat))
        except np.linalg.LinAlgError:
            print(an0)
            print(basis_mat)

    Y_rot = np.matmul(Y_curr, rot_mat.T)
    return Y_rot - np.mean(Y_rot)


def rotated_KL_min(V, X_tr, num_A, fit_skew=0):
    k = 3
    num_peaks = 10

    if V.shape[1] != X_tr.shape[0]:
        print(V.shape, X_tr.shape)
        print('Target & test set are not the same dimension!')

    ang_vals = np.linspace(0, np.pi, num_A)
    cos_g = np.cos(ang_vals)
    sin_g = np.sin(ang_vals)
    VL_r = []
    y = np.zeros(2 * num_A)
    for p in range(2 * num_A):
        pm = p % num_A
        ps = 2 * np.floor((p-1) / num_A) - 1
        rm = np.matmul([[ps, 0], [0, 1]], [[cos_g[pm], -sin_g[pm]], [sin_g[pm], cos_g[pm]]])

        if fit_skew != 0:
            sx = 0.2 + 0.2 * np.arange(1, 8)
            sy = 0.2 + 0.2 * np.arange(1, 8)

            ys = np.zeros(7 * 7)
            VL_rs = []

            for s1 in range(0, 7):
                for s2 in range(0, 7):
                    s_mat = [[sx[s1], 0], [0, sy[s2]]]
                    VL_rs.append(normal(V * rm * s_mat))
                    ys[s1 * 7 + s2] = eval_KL(VL_rs[s1 * 7 + s2], X_tr, k)

            y[p] = np.amin(ys)
            VL_r[p] = VL_rs[np.argmin(ys)]

        else:
            VL_r.append(normal(np.matmul(V, rm)))
            ys = eval_KL(VL_r[p], X_tr.T, k)
            y[p] = np.mean(ys)

    plt.plot(y)
    plt.xlabel('Rotation Angle')
    plt.ylabel('KL Divergence')
    plt.title('2D Rotation')
    plt.axvline(x=np.argmin(y))
    plt.show()

    V_out = VL_r[np.argmin(y)]

    peak_inds, peak_properties = find_peaks((np.amax(y) - y) / np.amax(y), height=0.0)

    peak_heights = peak_properties['peak_heights']

    descending_inds = np.argsort(peak_heights)[::-1]
    flip_inds = peak_inds[descending_inds]

    #V_flip = VL_r[flip_inds.tolist()]
    V_flip = []
    for i in range(len(peak_inds)):
        V_flip.append(VL_r[peak_inds[i]])

    return V_out, V_flip, y, flip_inds


def eval_KL(X, X_out, k=0):
    b_size = 50

    if k == 0:
        k0 = np.round(np.power(X.shape[1], 1 / 3))
        k1 = np.round(np.power(X_out.shape[1], 1 / 3))
    else:
        k0 = k1 = k


    if X.shape[1] == 3:
        p_train = prob_grid_3D(normal(X).T, b_size, k0)
        p_rot = prob_grid_3D(normal(X_out).T, b_size, k1)
    elif X.shape[1] == 2:
        p_train = prob_grid(normal(X).T, b_size, k0)
        p_rot = prob_grid(normal(X_out).T, b_size, k1)

    return error_nocenter(np.log(p_train), np.log(p_rot), X.shape[1])


def prob_grid_3D(X, b_size=50, k=0):
    return prob_grid(X[:, 0:2], b_size, k)


def prob_grid(X, b_size=50, k=0):
    w_size = 1

    X_n = X

    xy_max = np.amax(X_n) + w_size
    xy_min = np.amin(X_n) - w_size

    grid_axis = np.linspace(xy_min, xy_max, b_size)

    x1, y1 = np.meshgrid(grid_axis, grid_axis)

    return prob1(np.column_stack((x1.flatten('F'), y1.flatten('F'))), X_n.T, k)


def error_nocenter(p_train, p_rot, dim, num=1):
    if dim == 3:
        L = np.floor(np.power(p_train.shape[0], 1 / 3)).astype(int)
        pt = np.reshape(p_train, (L, L, L))
        pr = np.reshape(p_rot, (L, L, L))

        L_mid = np.floor(L / 2).astype(int)
        bd = np.zeros((L, L, L))
        bd[L_mid - num : L_mid + num, L_mid - num : L_mid + num, L_mid - num : L_mid + num] = 1

    elif dim == 2:
        L = np.floor(np.power(p_train.shape[0], 1 / 2)).astype(int)
        pt = np.reshape(p_train, (L, L))
        pr = np.reshape(p_rot, (L, L))

        L_mid = np.floor(L / 2).astype(int)
        bd = np.zeros((L, L))
        bd[L_mid - num: L_mid + num, L_mid - num: L_mid + num] = 1

    inds = bd != 1

    return np.linalg.norm(pt[inds] - pr[inds]) / np.linalg.norm(pr[inds])
