import numpy as np
import matplotlib.pyplot as plt


class ElliAlg:
    """
    The ElliAlg class represents an algorithm using ellipsoid method to solve convex optimization problems.
    """

    def __init__(self, instance) -> None:
        """
        Initializes an instance of the ElliAlg class.

        Parameters:
        - instance (dict): A dictionary containing the instance data.
        - a_max (float): Maximum acceleration.
        - y_init (float): Initial position.
        - lx (list): List of x-coordinates of the lower bound of the obstacle.
        - hx (list): List of x-coordinates of the upper bound of the obstacle.

        Returns:
        None
        """
        self.instance = instance
        self.a_max = instance['a_max']  # Maximum acceleration
        """Maximum acceleration."""
        self.y0 = instance['y_init']  # Initial position
        """Initial position."""
        self.lt = instance['lx']  # List of x-coordinates of the lower bound of the obstacle
        """List of x-coordinates of the lower bound of the obstacle."""
        self.ht = instance['hx']  # List of x-coordinates of the upper bound of the obstacle
        """List of x-coordinates of the upper bound of the obstacle."""
        self.T = len(self.ht) - 1  # Number of time steps
        """Number of time steps."""
        self.g = 9.8  # Acceleration due to gravity
        """Acceleration due to gravity."""
        self.opt_curve = [0.5e5]  # Optimal curve
        """Optimization curve."""
        self.n = self.T  # Number of time steps
        """Number of time steps."""
        self.eps = 1e-1  # Epsilon value
        """Epsilon value for stopping criterion."""
        self.a_opt = np.zeros(self.T) + 10  # Optimal acceleration
        """Initialize the optimal acceleration."""
        self.elli = Ellipsoid(self.a_opt, np.eye(self.T) * (self.T * self.a_max ** 2))  # Ellipsoid object
        """Ellipsoid object."""
        self.lower_bounds = []  # List of lower bounds
        """List of lower bounds."""
        self.elli_list = [self.elli]  # List of ellipsoids
        """List of ellipsoids."""
        pass

    def traj(self, a_vec):
        """
        Computes the trajectory given an acceleration vector.

        Parameters:
        - a_vec (array-like): The acceleration vector.

        Returns:
        - traj (list): The trajectory.
        """
        traj = [self.y0]
        for i in range(0, self.T):
            t = i + 1
            ct = self.subgrad_f1(t)
            traj.append(np.inner(ct, a_vec)+self.y0-t*(t-1)*self.g/2)
        return traj
    
    def energy(self, a_vec):
        """
        Computes the energy given an acceleration vector.

        Parameters:
        - a_vec (array-like): The acceleration vector.

        Returns:
        - energy (float): The energy.
        """
        return np.sum(np.ones(self.T)+a_vec+a_vec**2+a_vec**3)

    def fjt(self, a_vec):
        """
        Computes the constraint functions given an acceleration vector.

        Parameters:
        - a_vec (array-like): The acceleration vector.

        Returns:
        - fjt (array-like): The constraint functions.
        """
        f1t = np.zeros(self.T)
        f2t = np.zeros(self.T)
        f3t = np.zeros(self.T)
        f4t = np.zeros(self.T)
        for i in range(0, self.T):
            t = i + 1
            ct = self.subgrad_f1(t)
            f1t[i] = np.inner(ct, a_vec) + self.y0-t*(t-1)*self.g/2 - self.ht[t]
            f2t[i] = self.lt[t] - np.inner(ct, a_vec) - self.y0+t*(t-1)*self.g/2
            f3t[i] = a_vec[i] - self.a_max
            f4t[i] = -a_vec[i]
        return np.vstack([f1t, f2t, f3t, f4t])

    def subgrad_f0(self, a_vec):
        """
        Computes the subgradient of the objective function at a given acceleration vector.

        Parameters:
        - a_vec (array-like): The acceleration vector.

        Returns:
        - subgrad (array-like): The subgradient.
        """

        return 1+2*a_vec+3*a_vec**2

    def subgrad_fjt(self, j, t):
        """
        Computes the subgradient of the j-th constraint function at time t.

        Parameters:
        - j (int): The index of the constraint function.
        - t (int): The time step.

        Returns:
        - subgrad (array-like): The subgradient.
        """
        if j == 1:
            return self.subgrad_f1(t)
        elif j == 2:
            return self.subgrad_f2(t)
        elif j == 3:
            return self.subgrad_f3(t)
        elif j == 4:
            return self.subgrad_f4(t)
        else:
            raise print("j must in [1, 2, 3, 4]")

    def is_feasible(self, a_vec):
        """
        Checks if a given acceleration vector is feasible.

        Parameters:
        - a_vec (array-like): The acceleration vector.

        Returns:
        - is_feasible (bool or array-like): True if feasible, otherwise the indices of the violated constraints.
        """
        if np.all(self.fjt(a_vec) <= 0):
            return True
        else:
            return np.argwhere(self.fjt(a_vec) > 0)

    def update(self):
        """
        Updates the algorithm.

        Returns:
        - success (bool): True if the update was successful, False otherwise.
        """
        x = self.elli.x
        isFeasible = self.is_feasible(x)
        if isFeasible is True:
            g = self.subgrad_f0(x)
            f0 = self.energy(x)
            inner_prod = np.sqrt(np.inner(g, self.elli.P@g))
            self.lower_bounds.append(f0-inner_prod)
            if self.opt_curve[-1] > f0:
                self.opt_curve.append(f0)
                self.a_opt = x
            else:
                self.opt_curve.append(self.opt_curve[-1])
            if inner_prod <= self.eps:
                print("Suboptimal solution found, stopping")
                return False
            h = f0 - self.opt_curve[-1]
            self.elli_cut(g, h)
            return True
        else:
            j, t = isFeasible[0]
            g = self.subgrad_fjt(j+1, t+1)
            h = self.fjt(self.elli.x)[j, t]
            self.opt_curve.append(self.opt_curve[-1])
            self.lower_bounds.append(None)
            if h - np.sqrt(np.inner(g, self.elli.P@g)) > 0:
                print("Solution found at infeasible point, stopping")
                return False
            self.elli_cut(g, h)
            return True

    def elli_cut(self, g, h):
        """
        Performs an ellipsoid cut.

        Parameters:
        - g (array-like): The subgradient.
        - h (float): The constraint value.

        Returns:
        None
        """
        x = self.elli.x
        P = self.elli.P
        n = self.n
        alpha = h / np.sqrt(np.inner(g, P@g))
        g_hat = g / np.sqrt(np.inner(g, P@g))
        if alpha > 1:
            print("alpha > 1, intersection is empty")
            return
        else:
            x_ = x - ((1 + alpha*n)/(n + 1)) * (P @ g_hat)
            P_ = ((n**2*(1-alpha**2))/(n**2-1)) * (P-(2*(1+alpha*n)/((n+1)*(1+alpha)))*(P @ np.outer(g_hat, g_hat)) @ P)
            self.elli = Ellipsoid(x_, P_)

    def subgrad_f1(self, t):
        """
        Computes the subgradient of the first constraint function at time t.

        Parameters:
        - t (int): The time step.

        Returns:
        - subgrad (array-like): The subgradient.
        """
        return np.hstack([np.arange(t, 0, -1), np.zeros(self.T-t)])

    def subgrad_f2(self, t):
        """
        Computes the subgradient of the second constraint function at time t.

        Parameters:
        - t (int): The time step.

        Returns:
        - subgrad (array-like): The subgradient.
        """
        return -self.subgrad_f1(t)
    
    def subgrad_f3(self, t):
        """
        Computes the subgradient of the third constraint function at time t.

        Parameters:
        - t (int): The time step.

        Returns:
        - subgrad (array-like): The subgradient.
        """
        e_t = np.zeros(self.T)
        e_t[t-1] = 1
        return e_t
    
    def subgrad_f4(self, t):
        """
        Computes the subgradient of the fourth constraint function at time t.

        Parameters:
        - t (int): The time step.

        Returns:
        - subgrad (array-like): The subgradient.
        """
        e_t = np.zeros(self.T)
        e_t[t-1] = -1
        return e_t


class Ellipsoid:

    def __init__(self, x, P):
        """
        Initialize the class with given parameters.

        Parameters:
        - x (numpy.ndarray): The coordinate of the center of the ellipsoid.
        - P (numpy.ndarray): The matrix defining the ellipsoid, which should be positive definite.

        Returns:
        None
        """
        self.x = x
        """The coordinate of the center of the ellipsoid."""
        if np.linalg.det(P) < 0:
            print('P is not positive definite, det(P) = ', np.linalg.det(P))
        self.P = P
        """The matrix defining the ellipsoid, which should be positive definite."""

if __name__ == "__main__":
    import matplotlib.colors as mcolors
    from matplotlib.patches import Ellipse

    instance = np.load('data/32d_instance.npz', allow_pickle=False)
    elli_alg = ElliAlg(instance)
    print("a_max = ", elli_alg.a_max)
    print("y_init = ", elli_alg.y0)
    print("h_t = ", elli_alg.ht)
    print("l_t = ", elli_alg.lt)
    k = 0
    while elli_alg.update():
        k += 1
        if k % 100 == 0:
            print(f"Iteration {k}")
        if k > 1000:
            break
        elli_alg.elli_list.append(elli_alg.elli)
    print("Optimal value: ", elli_alg.opt_curve[-1])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(elli_alg.opt_curve[1:])
    ax[0,0].plot(elli_alg.lower_bounds, marker='.', ls='-', label='Lower bounds')
    ax[0,0].set_xlabel("Iteration")
    ax[0,0].set_ylabel("Objective value")
    ax[0,0].hlines(elli_alg.opt_curve[-1], 0, len(elli_alg.opt_curve)-1, colors='r', linestyles='dashed', label=f'Optimal value: {elli_alg.opt_curve[-1]}')
    
    ax[0,1].semilogy(elli_alg.opt_curve[1:]-elli_alg.opt_curve[-1])
    ax[0,1].set_xlabel("Iteration")
    ax[0,1].set_ylabel(r'$f(x_k)-f^*$')

    ax[1,1].plot(elli_alg.a_opt, marker='o', ls='--')
    ax[1,1].set_xlabel("Time")
    ax[1,1].set_ylabel("Acceleration")
    ax[1,1].hlines(elli_alg.a_max, 0, elli_alg.T, colors='r', linestyles='dashed', label=r'$a_{\rm max}$')
    ax[1,1].set_ylim([-1, elli_alg.a_max+1])
    ax[1,1].set_xlim([0-0.1, elli_alg.T-1+0.1])

    ax[1,0].plot(elli_alg.traj(elli_alg.a_opt), marker='.', ls='-')
    ax[1,0].set_xlabel("Time")
    ax[1,0].set_ylabel("Height")
    ax[1,0].plot(elli_alg.ht, ls='-', c='gray')
    ax[1,0].plot(elli_alg.lt, ls='-', c='gray')

    # Fill the area between maximum height and minimum height with gray color
    ax[1,0].fill_between(range(len(elli_alg.ht)), elli_alg.ht, 1000*np.ones(len(elli_alg.ht)), color='lightgray')
    ax[1,0].fill_between(range(len(elli_alg.lt)), -1000*np.ones(len(elli_alg.ht)), elli_alg.lt, color='lightgray',label='Barrier')
    ax[1,0].set_ylim([-1, elli_alg.ht.max()+10])
    #ax[1,0].set_xlim([0-0.1, elli_alg.T-1+0.1])

    ax[0,0].legend()
    ax[1,1].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    plt.show()

'''
#plot the ellispoids
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_xlabel(r"$a_0$")
ax.set_ylabel(r"$a_1$")
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['#ADD8E6', '#000080'])

for elli in elli_alg.elli_list:
    x = elli.x
    P = elli.P
    w, v = np.linalg.eig(P)
    theta = np.arctan2(v[1, 0], v[0, 0])
    a = np.sqrt(w[0])
    b = np.sqrt(w[1])
    e = Ellipse(xy=x, width=2*a, height=2*b, angle=theta*180/np.pi, alpha=0.1)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_facecolor(cmap(np.random.rand()))
    e.set_edgecolor('black')
plt.show()
'''
