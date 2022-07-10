import tensorflow as tf

def to_den(x):
    return tf.maximum(x*0.0+1e-6,x)

def to_den_mat(x):
    x += 1e-15 * tf.eye(tf.shape(x)[0],tf.shape(x)[1])
    return x

def gather_validate(value,idx,m):
    idx_ori = idx
    threshold = tf.cast(tf.shape(value)[0],idx.dtype)
    idx_valid = tf.where(tf.less(idx,threshold))
    idx_comp = tf.greater_equal(idx,threshold)
    idx_comp = tf.cast(idx_comp,tf.int64)
    
    idx = tf.gather(idx,idx_valid)
    idx = tf.reshape(idx,[-1])
    
    comp = tf.cast(tf.ones_like(idx_ori),tf.int64)
    comp = tf.cast(idx[-1] * comp,tf.int64)
    comp = tf.gather(comp,idx_comp)
    comp = tf.reshape(comp,[-1])
     
    idx = tf.concat([idx,comp[:m-tf.shape(idx)[0]]],0)
    idx = tf.reshape(idx,[-1])
    return idx

def del_idx(idx_src,idx_del):
    idx_src = tf.reshape(idx_src,[-1,1])
    idx_del = tf.reshape(idx_del,[1,-1])
    eq_mat = tf.to_float(tf.equal(idx_src,idx_del))
    idx_valid = tf.where(tf.equal(tf.reduce_sum(eq_mat,1),0))
    idx_src = tf.reshape(idx_src,[-1])
    idx_src = tf.gather(idx_src,idx_valid,axis=0)
    idx_src = tf.reshape(idx_src,[-1])
    return idx_src

class SimplexSolver:
    def __init__(self, n, m, A, b, c):
        self.n = n     # num of variables
        self.m = m     # num of constraints
        self.A = A     # constraints_left       (m,n) -> (m,n+m)
        self.b = b     # constrains_right       (m) -> (m)
        self.c = c     # optimization objective (n) -> (n+m)
        self.MAXITER = 10
        self.Laux_MAXITER = 5

    def initialize(self):
        # max obj
        self.c = tf.negative(self.c)
        # add slack or surplus variables =:0, >=:-1, <=:1 (in EPO, only = & >=)
        s_v = tf.negative(tf.eye(self.m))
        s_v = tf.concat([tf.zeros([1, self.m]), s_v[1:, :]], axis=0)
        self.A = tf.concat([self.A, s_v], axis=1)
        self.c = tf.concat([self.c, tf.zeros([self.m])],axis=0)
        # make b>=0
        for i in range(self.m):
            neg_A = tf.concat([self.A[:i,:],tf.reshape(tf.negative(self.A[i,:]),(1,tf.shape(self.A)[1])),self.A[i+1:,:]],axis=0)
            neg_b = tf.concat([self.b[:i],tf.reshape(tf.negative(self.b[i]),([1])),self.b[i+1:]],axis=0)
            def f_neg(): return neg_A, neg_b
            def f_keep(): return self.A, self.b
            self.A, self.b = tf.cond(tf.less(self.b[i],0), f_neg, f_keep)

    def auxPro(self, A_hat, b_hat):
        A = tf.identity(A_hat)     # (m,n+m)
        b = tf.identity(b_hat)     # (m)
        numPrecise = 1e-10
#         numPrecise = 0

        # record the col index of base variable: a[i]=k the ith entry of the kth column is 1, others 0
        baseVarList = tf.cast(tf.range(tf.shape(A)[1],tf.shape(A)[0]+tf.shape(A)[1]),dtype=tf.int64)   # (m)

        """
        Part I: Check the feasibility of the original problem
        """
        # add artificial variable
        auxMat = tf.eye(self.m)   # (m,m)
        A = tf.concat([A,auxMat],axis=1)  # (m,n+m+m)

        resVec = tf.concat([tf.zeros([tf.shape(A_hat)[1]]),tf.ones([tf.shape(A_hat)[0]])],axis=0)  # Contracted expenses (n+m+m)
        valFun = 0                        # obj value of aux LP

        for i in range(self.m):
            resVec = tf.subtract(resVec,A[i, :])     #(n+m+m)
            valFun = valFun - b[i]

        for i in range(self.Laux_MAXITER):
            minCol_id = tf.argmin(resVec)
            # assert tf.reduce_max(A[:, minCol_id]).eval() > 0, 'Problem no boundary!'
            b_temp = tf.where(tf.greater(A[:, minCol_id], 0), b / to_den(A[:, minCol_id]), tf.ones_like(b) * float('inf'))
            minRow_id = tf.argmin(b_temp)
            b_next = tf.concat(
                [b[:minRow_id], tf.reshape(b[minRow_id] / to_den(A[minRow_id, minCol_id]), ([1])), b[minRow_id + 1:]],
                axis=0)
            A_next = tf.concat(
                [A[:minRow_id, :], tf.reshape(A[minRow_id, :] / to_den(A[minRow_id, minCol_id]), (1, tf.shape(A)[1])),
                 A[minRow_id + 1:, :]], axis=0)  # (m,n+m+m)

            for rows in range(self.m):
                A_update = A_next[rows, :] - A_next[rows, minCol_id] * A_next[minRow_id, :]
                A_update = tf.concat([A_next[:rows, :], tf.reshape(A_update, (1, tf.shape(A)[1])), A_next[rows + 1:, :]],
                                     axis=0)
                b_update = b_next[rows] - A_next[rows, minCol_id] * b_next[minRow_id]
                b_update = tf.concat([b_next[:rows], tf.reshape(b_update, [1]), b_next[rows + 1:]], axis=0)

                def f1(): return A_update, b_update
                def f2(): return A_next, b_next

                A_next, b_next = tf.cond(tf.not_equal(tf.cast(rows, tf.int64), minRow_id), f1, f2)

            baseVarList_next = tf.concat(
                [baseVarList[:minRow_id], tf.reshape(minCol_id, ([1])), baseVarList[minRow_id + 1:]], axis=0)
            valFun_next = valFun - resVec[minCol_id] * b_next[minRow_id]
            resVec_next = resVec - resVec[minCol_id] * A_next[minRow_id, :]

            def f_loop(): return A_next, b_next, baseVarList_next, valFun_next, resVec_next
            def f_keep(): return A, b, baseVarList, valFun, resVec

            cond_next = tf.logical_or(tf.greater_equal(tf.reduce_min(resVec_next), 0),tf.less_equal(tf.abs(tf.reduce_min(resVec_next)), numPrecise))
            A, b, baseVarList, valFun, resVec = tf.cond(cond_next, f_keep, f_loop)    # if rj >= 0
        A, b, baseVarList, valFun, resVec = A_next, b_next, baseVarList_next, valFun_next, resVec_next

#         assert abs(valFun).eval() < numPrecise, 'Primal problem has no feasible solution'

        """
        ==============================================================================
        Part II: Remove redundant constraints, find BFS and its corresponding standard form
        """
        removed_idx = tf.where(tf.greater_equal(baseVarList, tf.cast(tf.shape(A_hat)[1],dtype=tf.int64)))
        for i in range(self.m):
            A_dell = tf.concat([A[:i, :], A[i + 1:, :]], axis=0)
            b_dell = tf.concat([b[:i],b[i+1:]],axis=0)

            idx_pivot = tf.where(tf.greater(tf.abs(A[i, :tf.shape(A_hat)[1]]), numPrecise))[0][0]
            b_pivot = tf.concat([b[:i],tf.reshape(b[i]/to_den(A[i,idx_pivot]),([1])),b[i+1:]],axis=0)
            A_pivot = tf.concat([A[:i,:],tf.reshape(A[i,:]/to_den(A[i,idx_pivot]),(1,tf.shape(A)[1])),A[i+1:,:]],axis=0)

            for rows in range(self.m):
                A_update = A_pivot[rows, :] - A_pivot[rows, idx_pivot] * A_pivot[i, :]
                A_update = tf.concat([A_pivot[:rows, :], tf.reshape(A_update, (1, tf.shape(A)[1])), A_pivot[rows + 1:, :]],axis=0)
                b_update = b_pivot[rows] - A_pivot[rows, idx_pivot] * b_pivot[i]
                b_update = tf.concat([b_pivot[:rows], tf.reshape(b_update, ([1])), b_pivot[rows + 1:]], axis=0)
                def f1(): return A_update, b_update
                def f2(): return A_pivot, b_pivot
                A_pivot, b_pivot = tf.cond(tf.not_equal(rows, i), f1, f2)
            baseVarList_pivot = tf.concat([baseVarList[:i], tf.reshape(idx_pivot,([1])), baseVarList[i + 1:]],axis = 0)

            def f_dell(): return A_dell, b_dell,baseVarList
            def f_pivot(): return A_pivot, b_pivot, baseVarList_pivot
            cond_del = tf.logical_and(tf.less_equal(tf.abs(tf.reduce_max(A[i,:tf.shape(A_hat)[1]])), numPrecise),tf.less_equal(tf.abs(tf.reduce_min(A[i,:tf.shape(A_hat)[1]])), numPrecise))
            A_del, b_del, baseVarList_del = tf.cond(cond_del, f_dell, f_pivot)

        def f_del(): return A_del, b_del, baseVarList_del
        def f_keep(): return A, b, baseVarList

        A, b, baseVarList = tf.cond(tf.greater(tf.shape(removed_idx)[0],0), f_del, f_keep)
 
        return A[:, :tf.shape(A_hat)[1]], tf.matrix_inverse(to_den_mat(tf.gather(A, baseVarList,axis=1))), b, baseVarList


    def revSimplex(self, A_hat, b_hat, c_hat, B_inverse_hat, baseVarList_hat):
        A = tf.identity(A_hat)                     # (m,n+m)
        b = tf.identity(b_hat)                     # (m)
        c = tf.identity(c_hat)                     # (n+m)
        B_inverse = tf.identity(B_inverse_hat)     # (m,m)
        baseVarList = tf.identity(baseVarList_hat) # (m)

        nonBaseVarList = tf.cast(tf.range(0,self.n+self.m),dtype=tf.int64) # (n)
#         for i in range(self.m):
#             idx = tf.where(tf.equal(nonBaseVarList,baseVarList[i]))[0][0]
#             nonBaseVarList = tf.concat([nonBaseVarList[:idx],nonBaseVarList[idx+1:]],axis=0)
            
        nonBaseVarList = del_idx(nonBaseVarList,baseVarList)

        b_line = tf.squeeze(tf.matmul(B_inverse, tf.reshape(b,(self.m,1))))  # (m)
        c_n = tf.gather(self.c, nonBaseVarList, axis=0)
        
        baseVarList=gather_validate(self.c, baseVarList,self.m)
        c_b = tf.gather(self.c, baseVarList, axis=0)
        
        mlambda = tf.squeeze(tf.matmul(tf.reshape(c_b,(1,self.m)), B_inverse))   # (m)
        N = tf.identity(tf.gather(A,nonBaseVarList,axis=1))

        numPrecise = 1e-10
#         numPrecise = 0

        for i in range(self.MAXITER):
            r = c_n - tf.squeeze(tf.matmul(tf.reshape(mlambda,(1,self.m)), N))
            #---------- next---------------------
#             idx_enter = tf.where(tf.logical_and(tf.less(r,0),tf.greater(tf.abs(r),numPrecise)))[0][0]

#             def f_normal(): return tf.where(tf.logical_and(tf.less(r,0),tf.greater(tf.abs(r),numPrecise)))[0][0]
#             def f_outlier(): return tf.where(tf.less_equal(r,0))[0][0]
        
            idx_enter = tf.cond(tf.greater(tf.reduce_sum(tf.to_float(tf.logical_and(tf.less(r,0),tf.greater(tf.abs(r),numPrecise)))),0), 
                                lambda: tf.logical_and(tf.less(r,0),tf.greater(tf.abs(r),numPrecise)), 
                                lambda: tf.less_equal(r,r))
        
            idx_enter = tf.where(idx_enter)[0][0]
        
#             idx_enter = tf.where(tf.less_equal(r,0))[0][0]
            q = nonBaseVarList[idx_enter]
            rq = r[idx_enter]
            yq = tf.squeeze(tf.matmul(B_inverse, tf.reshape(A[:, q],(self.m,1))))
            # assert tf.reduce_max(yq).eval() > 0, 'Problem no boundary!'
            b_temp = tf.ones_like(b_line) * float('inf')
            b_temp = tf.where(tf.greater(yq,numPrecise),b_line/to_den(yq),b_temp)
            p = tf.argmin(b_temp)

            mlambda_next = mlambda + rq / to_den(yq[p]) * B_inverse[p, :]

            Epq = tf.eye(self.m)
            Epq = tf.concat([Epq[:, :p], tf.reshape(tf.negative(yq / to_den(yq[p])), (self.m, 1)), Epq[:, p + 1:]], axis=1)
            one_hot = tf.one_hot(p, tf.shape(Epq)[0], dtype=tf.float32)
            new_value = one_hot * (1.0 / to_den(yq[p]))
            new_row = tf.reshape((Epq[p, :] - one_hot * Epq[p, p] + new_value), (1, self.m))
            Epq = tf.concat([Epq[:p, :], new_row, Epq[p + 1:, :]], 0)
            B_inverse_next = tf.matmul(Epq, B_inverse)

            nonBaseVarList_next = tf.where(tf.equal(nonBaseVarList, q), tf.fill(tf.shape(nonBaseVarList), baseVarList[p]),
                                      nonBaseVarList)
            nonBaseVarList_next = tf.reverse(tf.nn.top_k(nonBaseVarList_next, self.n, sorted=True)[0], [0])
            baseVarList_next = tf.where(tf.equal(baseVarList, baseVarList[p]), tf.fill(tf.shape(baseVarList), q), baseVarList)

            b_line_next = tf.squeeze(tf.matmul(B_inverse_next, tf.reshape(b,(self.m,1))))
            c_n_next = tf.gather(self.c, nonBaseVarList_next, axis=0)
            N_next = tf.identity(tf.gather(A, nonBaseVarList_next, axis=1))

            r_next = c_n_next - tf.squeeze(tf.matmul(tf.reshape(mlambda_next, (1, self.m)), N_next))
            #------------------------------------
            def f_keep(): return mlambda, B_inverse, nonBaseVarList, baseVarList, b_line, c_n, N
            def f_next(): return mlambda_next, B_inverse_next, nonBaseVarList_next, baseVarList_next, b_line_next, c_n_next, N_next

            optimal_cond = tf.logical_or(tf.greater_equal(tf.reduce_min(r_next),0), tf.less_equal(tf.abs(tf.reduce_min(r_next)),numPrecise))
            mlambda, B_inverse, nonBaseVarList, baseVarList, b_line, c_n, N = tf.cond(optimal_cond, f_keep, f_next)

        x_result = tf.zeros(tf.shape(A)[1])
        for i in range(self.m):
            x_result = tf.concat([x_result[:baseVarList_next[i]], tf.reshape(b_line_next[i],([1])), x_result[baseVarList_next[i] + 1:]],axis=0)

        valFun = tf.reduce_sum(tf.multiply(x_result, c))
        return valFun, x_result[:self.n]

    def solve(self):
        self.initialize()                                                  # initialize table
        Aaux, B_inverse, baux, baseVarList = self.auxPro(self.A, self.b)   # find base feasible solution and base variable
        rev_optimalVal, rev_result = self.revSimplex(Aaux, baux, self.c, B_inverse, baseVarList)  # solve the LP problem

        return rev_optimalVal, rev_result

# A = np.array([[1,2,0,-1,0,0],[3,0,2,0,1,0],[0,3,1,0,0,0]]).astype(float)
# b = np.array([4.,5.,6.])
# c= np.array([2.,5.,4., 0.,0.,0.])

# A = np.array([[2.,1.,1.,1.,0.,0.],[1.,2.,3.,0.,1.,0.],[2.,2.,1.,0.,0.,1.]])
# b = np.array([2.,5.,6.])
# c= np.array([-3.,-1.,-3., 0.,0.,0.])

# A = np.array([[6,1,-2,-1,0,0],[1,1,1,0,1,0],[6,4,-2,0,0,-1]]).astype(float)
# b = np.array([5.,4.,10.])
# c= np.array([5.,2.,-4., 0.,0.,0.])

# with tf.Session() as sess:
#     sp = SimplexSolver(3, 3, A, b, c)
#     value, alpha = sp.solve()
#     print(value.eval())
#     print(alpha.eval())