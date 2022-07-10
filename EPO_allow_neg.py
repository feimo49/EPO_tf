import numpy as np
import tensorflow as tf
from Simplex import SimplexSolver

def neg(x):
    return tf.minimum(x*0.0,x)

def to_den(x):
    return tf.maximum(x*0.0+1e-6,x)

class EPO_LP:
    def __init__(self, m, n, r, eps=1e-4):
        self.m = m          # task num
        self.n = n          # parameter num
        self.r = r          # reference vector
        self.eps = eps      # threshold
        self.last_move = None
        self.gamma = 0      # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0      # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
#         assert tf.shape(l)[1] == tf.shape(G)[1] == tf.shape(r)[1] == self.m, "length != m"
        rl, self.mu_rl, a = self.adjustments(l, r)       # rl : (1,m), mu_rl : float, a : (1,m),Adjustments
        C = G if C else tf.matmul(G, tf.transpose(G))    # tensor (m,m) : Gradient inner products, G^T G
        Ca = tf.matmul(a, tf.transpose(C))               # tensor (1,m) : d_bal^TG

        # objective for Problems
        obj_balance = tf.identity(Ca)                    # tensor (1,m) : obj for balance, Maximize(alpha @ Ca)
        obj_dsecent = tf.reshape(tf.reduce_sum(tf.identity(C), 1), (1, self.m))  # tensor (1,m) : obj for descent, Maximize(sum(alpha @ C))

        rhs = tf.identity(Ca)                            # tensor (1,m): RHS of constraints for balancing
        J = Ca > 0
        J_star_idx = tf.equal(rl,tf.reduce_max(rl))
        rhs = tf.where(J,tf.fill(tf.shape(rhs),tf.cast(tf.negative(0x3f3f3f3f),dtype = tf.float32)),rhs) # Not efficient; but works.
        rhs = tf.where(J_star_idx,tf.zeros_like(rhs,dtype=tf.float32),rhs)
        rhs = tf.cond(tf.greater((tf.shape(tf.where(J)))[0],0), lambda: tf.identity(rhs),lambda: tf.zeros_like(Ca))

        # constraints for Problems
        num_cons = self.m  +1
        # balance (Simplex)
        constraints_left_balance = tf.concat([ tf.ones([1, self.m]), C,tf.ones([1, self.m])], axis=0,name='EPO_concat_l_38')                         # sum(self.alpha) == 1
        constraints_right_balance = tf.concat([ tf.ones([1, 1]), rhs,tf.ones([1, 1])], axis=1,name='EPO_concat_l_39')                           # C @ alpha >= rhs
        # descent (Relaxed)
        constraints_left_descent_relaxed = tf.concat([ tf.ones([1, self.m]), C,tf.ones([1, self.m])], axis=0,name='EPO_concat_l_41')                 # sum(self.alpha) == 1
        constraints_right_descent_relaxed = tf.concat([ tf.ones([1, 1]), tf.zeros_like(rhs),tf.ones([1, 1])], axis=1,name='EPO_concat_l_42')    # C @ alpha >= 0
        # descent (Restricted)
        constraints_left_descent_restricted = tf.concat([ tf.ones([1, self.m]), C,Ca], axis=0,name='EPO_concat_l_44')  # alpha @ Ca >= -neg(np.max(Ca))
        constraints_right_descent_restricted = tf.concat([ tf.ones([1, 1]), tf.zeros_like(rhs), tf.reshape(-neg(tf.reduce_max(Ca)),(1,1))], axis=1,name='EPO_concat_l_45')

        def f_relax(): return constraints_left_descent_relaxed,constraints_right_descent_relaxed, num_cons
        def f_restricted(): return constraints_left_descent_restricted,constraints_right_descent_restricted, num_cons+1
        constraints_left_descent, constraints_right_descent, num_cons_descent = tf.cond(tf.equal(relax,True),f_relax,f_restricted)

        # LP balance
        def f_balance():
            self.last_move = "bal"
            return obj_balance,constraints_left_balance,constraints_right_balance, num_cons

        # LP dominance
        def f_descent():
            self.last_move = "dom"
            return obj_dsecent, constraints_left_descent, constraints_right_descent, num_cons_descent

          
        obj,constraints_left,constraints_right, self.num_cons = tf.cond(tf.greater(self.mu_rl,self.eps), f_balance, f_descent)
        prob = SimplexSolver(self.m, self.m +2, constraints_left,tf.squeeze(constraints_right), tf.squeeze(obj))
        self.gamma, alpha = prob.solve()

        return alpha/to_den(tf.reduce_sum(alpha))

    def mu(self, rl, normed=False):
#         assert tf.equal(tf.shape(tf.where(rl < 0))[0], 0), "rl<0 \n rl={}".format(rl)
        m = tf.shape(rl)[1]
        m = tf.cast(m, dtype=tf.float32)
        l_hat = rl if normed else rl / (tf.reduce_sum(rl)+1e-6)
        l_hat = tf.where(tf.equal(l_hat,0), tf.fill(tf.shape(l_hat),1e-6),l_hat)
        return tf.reduce_sum(l_hat * tf.log(l_hat * m))


    def adjustments(self, l, r=1):
        m = tf.shape(l)[1]
        m = tf.cast(m, dtype=tf.float32)
        rl = r * l
        l_hat = rl / tf.reduce_sum(rl)
        mu_rl = self.mu(l_hat, normed=True)
        a = r * (tf.log(l_hat * m + 1e-6) - mu_rl)
        return rl, mu_rl, a


# if __name__ == "__main__":
 
#     m = 3                                # int : task nums
#     n = 100                              # int : parameter nums
#     r = tf.reshape(tf.constant([3.,2.,1.]),[1,m])                   # tensor (1,m) : preference vector
#     r /= tf.reduce_sum(r)
#     epo_lp = EPO_LP(m,n,r)
#     losses = tf.reshape(tf.constant([0.15,0.42,2.30]),(1,m))   # tensor (1,m) : losses of different tasks
#     G = tf.constant([[0.3,0.2,0.1,0.6],[-0.3,-0.2,-0.1,-0.6],[0.1,0.0,-0.3,0.0]])
#     GG = tf.matmul(G,G,transpose_b=True) 

#     with tf.Session() as sess:
#         alpha = epo_lp.get_alpha(losses, G=GG, C=True)
#         print(sess.run(alpha))


