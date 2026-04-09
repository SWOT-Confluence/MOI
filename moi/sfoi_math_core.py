import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import warnings
from scipy.stats.distributions import chi2

def calcG_sparse(m, n, junctions, basin_dict, reachlist='reach_ids_all'):
    rows, cols, data = [], [], []
    reach_map = {str(r): i for i, r in enumerate(basin_dict[reachlist])}
    
    for junction in junctions:
        row = junction['row_num']
        for upflow in junction['upflows']:
            if str(upflow) in reach_map:
                rows.append(row)
                cols.append(reach_map[str(upflow)])
                data.append(1)
        for downflow in junction['downflows']:
            if str(downflow) in reach_map:
                rows.append(row)
                cols.append(reach_map[str(downflow)])
                data.append(-1)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, n))

def getA_sparse(C_sparse, n, D, l):
    I_n = sp.eye(n, format='csr')
    A = sp.bmat([[C_sparse], [I_n]], format='csr')
    A_lil = A.tolil()
    Dind = np.array([D[l+i] for i in range(n)])
    
    for i in range(l):
        nonzero_cols = A_lil.rows[i]
        if not nonzero_cols: continue
        sum_Dind = np.sum(Dind[nonzero_cols])
        if sum_Dind > 0:
            scale = D[i] / sum_Dind
            for idx in range(len(nonzero_cols)):
                A_lil.data[i][idx] *= scale
    return A_lil.tocsr()

def getP_1D(covQ, Qbar, norm=2):
    σQ = np.abs(np.array(Qbar)) * covQ
    σQ = np.clip(σQ, 1e-6, np.inf) 
    return 1.0 / (σQ ** norm), 1.0 / (σQ ** (norm / 2.0))

def adjust_lsq_sparse(A_sparse, W_1d, L, o=1, bound=True, idxbad=None):
    if np.any(np.isnan(W_1d)):
        raise ValueError("DEBUG: W_1d contains NaN!")
    if np.any(np.isnan(L)):
        raise ValueError("DEBUG: L (Observation data) contains NaN!")
    if np.any(np.isnan(A_sparse.data)):
        raise ValueError("DEBUG: A_sparse contains NaN!")
    
    mA, nA = A_sparse.shape
    m = mA - o
    
    x = cp.Variable(nA)
    residual = cp.multiply(W_1d, A_sparse @ x - L)
    objective = cp.Minimize(cp.sum_squares(residual))
    
    constraints = []
    if bound:
        LB = np.zeros(mA)
        constraints = [A_sparse @ x >= LB]
        # constraints = [x >= 0]
        
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.OSQP, max_iter=100000, eps_abs=1e-4, eps_rel=1e-4, adaptive_rho=True, verbose=False)

        if prob.status != cp.OPTIMAL:
            print(f"OSQP status is '{prob.status}'. Triggering SCS fallback...")
            raise ValueError("OSQP did not reach strictly OPTIMAL status.")
            
        if x.value is None:
            raise ValueError(f"OSQP returned None. Problem status: {prob.status}")
        return x.value, "success"
        
    except Exception as e:
        print(f"OSQP failed ({str(e)}). Switching to SCS solver...")
        try:
            prob.solve(verbose=True, solver=cp.SCS, max_iters=50000)
            if x.value is None:
                raise ValueError(f"SCS returned None. Problem status: {prob.status}")

            if prob.status == cp.OPTIMAL_INACCURATE:
                print("WARNING: SCS also returned 'optimal_inaccurate', accepting dirty solution to proceed.")
                
            return x.value, "success_scs"
        except Exception as e2:
            raise RuntimeError(f"Solver completely failed. OSQP error: {str(e)} | SCS error: {str(e2)}")
            
            
            
            
def adjust_lsq_sparse_irls(A_sparse, W_1d, L, o=1, bound=True, idxbad=None, itermax=15, limit=2.5):

    mA, nA = A_sparse.shape
    dof = mA - nA
    if dof <= 0: dof = 1 
    
    W = np.copy(W_1d)
    P = W**2
    cst_pass = False
    iters = 0
    eps = np.finfo(float).eps
    
    filter_mask = np.full((mA,), True)
    s = np.zeros((mA,))
    
    x_best = None
    status_best = "failed"
    
    while not cst_pass and iters <= itermax:
        try:
            x, status = adjust_lsq_sparse(A_sparse=A_sparse, W_1d=W, L=L, o=o, bound=bound, idxbad=idxbad)
            x_best = x
            status_best = status
        except Exception as e:
            print(f"      [IRLS] Solver failed at iter {iters}: {e}")

        V = L - (A_sparse @ x)  # 残差
        
        So = np.sqrt(np.dot(V, P * V) / dof)
        X = (So**2) * dof
        
        factor = 1 / W * So

        s[filter_mask] = np.abs(V[filter_mask] / factor[filter_mask])
        filter_mask = s <= limit
        
        X1 = chi2.ppf(1 - 0.05 / 2, dof)
        X2 = chi2.ppf(0.05 / 2, dof)
        
        if X < X2 or X > X1:
            cst_pass = False
            f = np.full(np.shape(L), 1.0)
            outlier_idx = s > limit
            f[outlier_idx] = 10**(limit - s[outlier_idx])
            
            f = np.clip(f, eps, np.inf)
            minvalue = 2 * np.sqrt(eps)
            
            P = (f / factor)**2
            W = f / factor
            
            W = np.clip(W, minvalue, np.inf)
            P = np.clip(P, minvalue, np.inf)
            
            if idxbad is not None:
                W[idxbad] = minvalue
                P[idxbad] = minvalue
        else:
            cst_pass = True
            # print(f'      [IRLS] Passed chi2 test with So={So:.4f} at iter {iters}')
            
        iters += 1

    return x_best, status_best
    
    
def adjust_lsq_sparse_strict_mass(A_sparse, W_1d, L, n_mass_rows, bound=True):

    mA, nA = A_sparse.shape
    n_prior_rows = mA - n_mass_rows
    
    A_prior = A_sparse[:n_prior_rows, :]
    A_mass = A_sparse[n_prior_rows:, :]
    
    W_prior = W_1d[:n_prior_rows]
    L_prior = L[:n_prior_rows]
    
    x = cp.Variable(nA)
    

    residual = cp.multiply(W_prior, A_prior @ x - L_prior)
    objective = cp.Minimize(cp.sum(cp.huber(residual, M=2.0)))

    constraints = []
    if bound:
        constraints.append(A_prior @ x >= 0)  
        
    constraints.append(A_mass @ x == 0)
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS, max_iters=50000, verbose=False)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return x.value, "success"
        else:
            prob.solve(solver=cp.OSQP, max_iter=50000)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return x.value, "success_osqp"
            else:
                raise ValueError(f"Status {prob.status}")
                
    except Exception as e:
        print(f"      [Strict Mass Solver] failed: {e}")
        return None, "failed"