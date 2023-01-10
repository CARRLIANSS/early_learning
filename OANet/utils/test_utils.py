import cv2
from utils.transformations import quaternion_from_matrix
import numpy as np
from multiprocessing import Pool as ThreadPool
import os
from six.moves import xrange

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t

def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(E_hat)
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
            #import pdb;pdb.set_trace()
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


def eval_decompose(p1s, p2s, dR, dt, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        if probs is None and method != "MLESAC":
            # Change the threshold from 0.01 to 0.001 can largely imporve the results
            # For fundamental matrix estimation evaluation, we also transform the matrix to essential matrix.
            # This gives better results than using findFundamentalMat
            E, mask_new = cv2.findEssentialMat(p1s_good, p2s_good, method=method, threshold=0.001)

        else:
            pass
        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t

def dump_res(measure_list, res_path, eval_res, tag):
    # dump test results
    for sub_tag in measure_list:
        # For median error
        ofn = os.path.join(res_path, "median_{}_{}.txt".format(sub_tag, tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(eval_res[sub_tag])))

    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    t_acc_hist, _ = np.histogram(cur_err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)
    # Store return val
    for _idx_th in xrange(1, len(ths)):
        ofn = os.path.join(res_path, "acc_q_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_t_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_qt_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    ofn = os.path.join(res_path, "all_acc_qt_auc20_{}.txt".format(tag))
    np.savetxt(ofn, np.maximum(cur_err_q, cur_err_t))
    ofn = os.path.join(res_path, "all_acc_q_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_q)
    ofn = os.path.join(res_path, "all_acc_t_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_t)

    # Return qt_auc20
    ret_val = np.mean(qt_acc[:4])  # 1 == 5
    return ret_val

def denorm(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x

def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res

def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config, = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)
    else:
        # actually not use prob here since probs is None
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=cv2.RANSAC, \
            probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3,3)
        _t_hat = np.random.randn(3,1)
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]