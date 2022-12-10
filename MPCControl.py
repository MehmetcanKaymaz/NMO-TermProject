import casadi as ca
import numpy as np
from scipy import sparse


class NMPCComplex:
    def __init__(self, dt=0.01,N=20):
        self.info = "Nonlinear Model Predictive Control for Simple Vehicle Models"

        self._dt = dt
        self._N = N

        self._x_dim = 5
        self._u_dim = 2

        self._Q = sparse.diags([1,1,0,0,0])
        self._R = sparse.diags([.1,.1])

        self._x0 = np.zeros(5)
        self._u0 = 0

        self._initDynamics()

    def _initDynamics(self,):
        x,y,v,theta,q = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('v'),ca.MX.sym('theta'),ca.MX.sym('q')

        self._x = ca.vertcat(x,y,v,theta,q)

        u1,u2=ca.MX.sym('u1'), ca.MX.sym('u2')
        self._u= ca.vertcat(u1,u2)

        x_dot = ca.vertcat(v*ca.cos(theta),
                           v*ca.sin(theta),
                           u1,
                           q,
                           u2)

        self.f = ca.Function('f', [self._x, self._u], [
                             x_dot], ['x', 'u'], ['ode'])

        intg_options = {'tf': .01, 'simplify': True,
                        'number_of_finite_elements': 4}
        dae = {'x': self._x, 'p': self._u, 'ode': self.f(self._x, self._u)}

        intg = ca.integrator('intg', 'rk', dae, intg_options)

        Delta_x = ca.SX.sym("Delta_x", self._x_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        cost_track = Delta_x.T @ self._Q @ Delta_x
        cost_u = Delta_u.T @ self._R @ Delta_u

        f_cost_track = ca.Function('cost_track', [Delta_x], [cost_track])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        x_next = intg(x0=self._x, p=self._u)['xf']
        F = ca.Function('F', [self._x, self._u], [
                        x_next], ['x', 'u'], ['x_next'])

        umin = [-10,-np.pi]
        umax = [10,np.pi]
        xmin = [-ca.inf,-ca.inf,-10,-ca.inf,-np.pi]
        xmax = [ca.inf,ca.inf,10,ca.inf,np.pi]

        opti = ca.Opti()

        X = opti.variable(self._x_dim, self._N+1)
        self.P = opti.parameter(self._x_dim, 2)
        self.U = opti.variable(self._u_dim, self._N)

        cost = 0

        opti.subject_to(X[:, 0] == self.P[:, 0])

        #cost += f_cost_track((X[:, self._N]-self.P[:, 1]))
        for k in range(self._N):
            cost += f_cost_track((X[:, k]-self.P[:, 1]))
            if k == 0:
                cost += 0
            else:
                cost += f_cost_u((self.U[:, k]-self.U[:, k-1]))

            opti.subject_to(X[:, k+1] == F(X[:, k], self.U[:, k]))
            opti.subject_to(self.U[:, k] <= umax)
            opti.subject_to(self.U[:, k] >= umin)
            opti.subject_to(X[:, k+1] >= xmin)
            opti.subject_to(X[:, k+1] <= xmax)
            #opti.subject_to((X[:2, self._N]-self.P[:2, 1]).T@(X[:2, self._N]-self.P[:2, 1])<=.1)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 1000,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
        }


        opti.minimize(cost)
        opti.solver('ipopt', ipopt_options)
 
        self.opti = opti

    def run_controller(self, x0, ref_states):

        p = np.concatenate((x0.reshape((self._x_dim, 1)), ref_states), axis=1)

        self.opti.set_value(self.P, p)
        sol = self.opti.solve()
        u = sol.value(self.U)
        return u


class NMPCBase:
    def __init__(self, dt=0.01,N=20,V=10):
        self.info = "Nonlinear Model Predictive Control for Simple Vehicle Models"

        self._dt = dt
        self._N = N
        self.V=V

        self._x_dim = 3
        self._u_dim = 1

        self._Q = sparse.diags([1,1,0])
        self._R = .1

        self._x0 = np.zeros(3)
        self._u0 = 0

        self._initDynamics()

    def _initDynamics(self,):
        x,y,theta = ca.MX.sym('x'), ca.MX.sym('y'),ca.MX.sym('theta')

        self._x = ca.vertcat(x,y,theta)

        self._u=ca.MX.sym('u')

        x_dot = ca.vertcat(self.V*ca.cos(theta),
                           self.V*ca.sin(theta),
                           self._u,)

        self.f = ca.Function('f', [self._x, self._u], [
                             x_dot], ['x', 'u'], ['ode'])

        intg_options = {'tf': .01, 'simplify': True,
                        'number_of_finite_elements': 4}
        dae = {'x': self._x, 'p': self._u, 'ode': self.f(self._x, self._u)}

        intg = ca.integrator('intg', 'rk', dae, intg_options)

        Delta_x = ca.SX.sym("Delta_x", self._x_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        cost_track = Delta_x.T @ self._Q @ Delta_x
        cost_u = Delta_u.T @ self._R @ Delta_u

        f_cost_track = ca.Function('cost_track', [Delta_x], [cost_track])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        x_next = intg(x0=self._x, p=self._u)['xf']
        F = ca.Function('F', [self._x, self._u], [
                        x_next], ['x', 'u'], ['x_next'])

        umin = -np.pi
        umax = np.pi
        xmin = [-ca.inf,-ca.inf,-ca.inf]
        xmax = [ca.inf,ca.inf,ca.inf]

        opti = ca.Opti()

        X = opti.variable(self._x_dim, self._N+1)
        self.P = opti.parameter(self._x_dim, 2)
        self.U = opti.variable(self._u_dim, self._N)

        cost = 0

        opti.subject_to(X[:, 0] == self.P[:, 0])

        #cost += f_cost_track((X[:, self._N]-self.P[:, 1]))
        for k in range(self._N):
            cost += f_cost_track((X[:, k]-self.P[:, 1]))
            if k == 0:
                cost += 0
            else:
                cost += f_cost_u((self.U[:, k]-self.U[:, k-1]))

            opti.subject_to(X[:, k+1] == F(X[:, k], self.U[:, k]))
            opti.subject_to(self.U[:, k] <= umax)
            opti.subject_to(self.U[:, k] >= umin)
            opti.subject_to(X[:, k+1] >= xmin)
            opti.subject_to(X[:, k+1] <= xmax)
            #opti.subject_to((X[:2, self._N]-self.P[:2, 1]).T@(X[:2, self._N]-self.P[:2, 1])<=.1)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 1000,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
        }


        opti.minimize(cost)
        opti.solver('ipopt', ipopt_options)
 
        self.opti = opti

    def run_controller(self, x0, ref_states):

        p = np.concatenate((x0.reshape((self._x_dim, 1)), ref_states), axis=1)

        self.opti.set_value(self.P, p)
        sol = self.opti.solve()
        u = sol.value(self.U)
        return u
