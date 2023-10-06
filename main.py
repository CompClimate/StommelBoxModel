import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pprint import pprint


plt.rcParams.update({
    'text.usetex': True,
})


YEAR = 365 * 24 * 3600
Sv = 1.e9 # m^3/sec


class BoxModel:
    def __init__(self, S0, S1, S2, T0, T1, T2, alpha, beta, k, area, depth):
        self.S0 = S0
        self.S1 = S1
        self.S2 = S2
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.area = area
        self.depth = depth

    @property
    def DeltaT(self):
        return self.T1 - self.T2

    @property
    def DeltaS(self):
        return self.S1 - self.S2
    
    @property
    def V(self):
        return self.area * self.depth

    def q(self, DeltaT, DeltaS):
        """Implements the AMOC volume transport q."""
        # THC transport in m^3/sec as function of temperature and salinity
        # difference between the boxes
        
        rho_1_2_diff = self.alpha * DeltaT - self.beta * DeltaS
        flow = self.k * rho_1_2_diff
        
        return flow

    def d_q_t(self, F):
        """Implements the analytical derivative of q wrt. t."""
        return -self.k * self.beta * 2 * (F - abs(self.q(self.DeltaT, self.DeltaS)) * self.DeltaS)

    def d_DeltaS_t(self, F, DeltaT, DeltaS):
        """Implementation according to Potsdam"""
        return -2 * abs(self.q(DeltaT, DeltaS)) * DeltaS - 2 * F

    def rho(self, T, S):
        return self.rho0 * (1 - self.alpha * T + self.beta * S)
    
    def steady_states(self, F, X):
        """Implements the steady state solutions of the model."""
        Y = np.zeros((3,))
    
        y1 = (X - np.sqrt(X**2 + 4 * self.beta * F / self.k)) / 2
        val = X**2 - 4 * self.beta * F / self.k
    
        # The argument to the square root function has to be positive.
        if val > 0:
            offset = X / 2
            y2 = offset + np.sqrt(val) / 2
            y3 = offset - np.sqrt(val) / 2
        # Simply set the solution to NaN if the argument is non-positive.
        else:
            y2 = np.nan
            y3 = np.nan
        
        Y = np.array([y1, y2, y3])
        return Y

    def Fs_func(self, time, time_max, is_Fs_time_dependent):
        # total surface salt flux into northern box
        # Specify maximum and minimum of freshwater forcing range during the
        # run in m/year:
        FW_min = -0.1
        FW_max = 5
    
        if is_Fs_time_dependent:
            flux = FW_min + (FW_max - FW_min) * time / time_max
        else:
            flux = 2
    
        # convert to total salt flux:
        return flux * self.area * self.S0 / YEAR

    def rhs_S(self, time, time_max, is_Fs_time_dependent, DeltaT=None, DeltaS=None):
        # Input: q in m^3/sec; FW is total salt flux into box
        DeltaT = self.DeltaT if DeltaT is None else DeltaT
        DeltaS = self.DeltaS if DeltaS is None else DeltaS
        
        F_s = self.Fs_func(time, time_max, is_Fs_time_dependent)
        # rhs = 2 * (F_s - abs(self.q(DeltaT, DeltaS)) * DeltaS)
        # Optional: with S0
        # rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaS + 2 * F_s * self.S0)
        rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaS + 2 * F_s)
        return rhs / self.V

    def simulate(self, Fs_range):
        # Compute the steady state solutions as well as the q values for the above values of F.
        DeltaS_steady = np.zeros((3, len(Fs_range)))
        q_steady = np.zeros((3, len(Fs_range)))
        
        for i, Fs in enumerate(Fs_range):
            Y = self.steady_states(Fs, self.alpha * self.DeltaT)
            # translate Y solution to a solution for DeltaS:
            DeltaS_steady[:, i] = Y / self.beta
            for j in range(3):
                q_steady[j, i] = self.q(self.DeltaT, DeltaS_steady[j, i])

        return DeltaS_steady, q_steady


def scale_Fs(Fs, area, S0, year):
    return Fs * area * S0 / year


def Fs_to_m_per_year(S0, area):
    return S0 * area / YEAR


def plot_steady(Fs_range, DeltaS_steady, q_steady, S0, area, normalize=True):
    plt.figure(1, figsize=(8, 4), dpi=200)
    
    Fs_range_normalized = np.copy(Fs_range)
    if normalize:
        Fs_range_normalized /= Fs_to_m_per_year(S0, area, YEAR)
    
    plt.subplot(1, 2, 1)
    # plot all three solutions for Delta S as function of Fs in units of m/year:
    plt.plot(Fs_range_normalized, DeltaS_steady[0, :], 'r.', markersize=1)
    plt.plot(Fs_range_normalized, DeltaS_steady[1, :], 'g.', markersize=1)
    plt.plot(Fs_range_normalized, DeltaS_steady[2, :], 'b.', markersize=1)
    
    # plot a dash think line marking the zero value:
    plt.plot(Fs_range_normalized, np.zeros(DeltaS_steady.shape[1]), 'k--', dashes=(10, 5), linewidth=0.5)
    plt.title('(a) steady states')
    plt.xlabel('$F_s$ (m/year)');
    plt.ylabel('$\Delta S$');
    plt.xlim([min(Fs_range_normalized), max(Fs_range_normalized)])
    
    plt.subplot(1, 2, 2)
    # plot all three solutions for q (in Sv) as function of Fs in units of m/year:
    plt.plot(Fs_range_normalized, q_steady[0, :], 'r.', markersize=1)
    plt.plot(Fs_range_normalized, q_steady[1, :], 'g.', markersize=1)
    plt.plot(Fs_range_normalized, q_steady[2, :], 'b.', markersize=1)
    plt.plot(Fs_range_normalized, np.zeros(q_steady.shape[1]), 'k--', dashes=(10, 5), linewidth=0.5)
    plt.title('(b) steady states')
    plt.xlabel('$F_s$ (m/year)')
    plt.ylabel('$q$ (Sv)')
    plt.tight_layout()

    ls_min_max_1 = [(min(DeltaS_steady[i, :]), max(DeltaS_steady[i, :])) for i in range(3)]
    ls_min_max_2 = [(min(q_steady[i, :]), max(q_steady[i, :])) for i in range(3)]

    return ls_min_max_1, ls_min_max_2


def plot_rhs(model, time, time_max, DeltaS_range, fig=None, ax=None):
    is_Fs_time_dependent = False
    DeltaS_range = np.arange(-3, 0, 0.01)
    rhs = np.zeros(len(DeltaS_range))

    for i in range(len(DeltaS_range)):
        DeltaS = DeltaS_range[i]
        rhs[i] = model.rhs_S(time, time_max, is_Fs_time_dependent, DeltaS=DeltaS)

    rhs /= np.std(rhs)

    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    
    ax.plot(DeltaS_range, rhs, 'k-', lw=2)
    ax.plot(DeltaS_range, rhs * 0, 'k--', dashes=(10, 5), lw=0.5)

    # superimpose color markers of the 3 solutions
    Fs = model.Fs_func(0.0, 0.0, False)
    yy = model.steady_states(Fs, model.alpha * model.DeltaT) / model.beta

    ax.plot(yy[0], 0, 'ro', markersize=10)
    ax.plot(yy[1], 0, 'go', markersize=10)
    ax.plot(yy[2], 0, 'bo', markersize=10, fillstyle='none')
    ax.set_title('(c) stability')
    ax.set_xlabel('\(\Delta S\)')
    ax.set_ylabel('\(d\Delta S/dt\)')

    return fig, ax


def plot_time_series(model, time_max, fig=None, ax=None):
    is_Fs_time_dependent = True
    teval1 = np.arange(0, time_max, time_max / 1000)
    teval2 = np.arange(time_max / 2, time_max, time_max / 1000)
    tspan1 = (teval1[0], teval1[-1])
    tspan2 = (teval2[0], teval2[-1])
    
    sol_DS1 = solve_ivp(
        fun=lambda time, DeltaS: model.rhs_S(time, time_max, is_Fs_time_dependent, DeltaS=DeltaS),
        vectorized=False,
        y0=[0],
        t_span=tspan1,
        t_eval=teval1,
        dense_output=True,
    )
    Time_DS1 = sol_DS1.t
    DeltaS1 = sol_DS1.y

    sol_DS2 = solve_ivp(
        fun=lambda time, DeltaS: model.rhs_S(time, time_max, is_Fs_time_dependent, DeltaS=DeltaS),
        vectorized=False,
        y0=[DeltaS1[0][-1]],
        t_span=tspan2,
        t_eval=teval2,
        dense_output=True,
    )
    Time_DS2 = sol_DS2.t
    DeltaS2 = sol_DS2.y

    is_Fs_time_dependent = True
    FWplot1 = np.zeros(len(Time_DS1))
    qplot1 = np.zeros(len(Time_DS1))

    for i, t in enumerate(Time_DS1):
        FWplot1[i] = model.Fs_func(t, time_max, is_Fs_time_dependent)
        qplot1[i] = model.q(model.DeltaT, DeltaS1[0, i])

    FWplot2 = np.zeros(len(Time_DS1))
    qplot2 = np.zeros(len(Time_DS1))

    for i, t in enumerate(Time_DS2):
        FWplot2[i] = model.Fs_func(t, time_max, is_Fs_time_dependent)
        qplot2[i] = model.q(model.DeltaT, DeltaS2[0, i])

    xs1 = Time_DS1 / YEAR / 1000
    xs2 = Time_DS2 / YEAR / 1000

    xs = np.append(xs1, xs2)
    Time_DS = np.append(Time_DS1, Time_DS2)
    qplot = np.append(qplot1, qplot2) / Sv
    DeltaS = np.append(DeltaS1, DeltaS2, axis=1)

    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(figsize=(9, 7))

    artists = []
    a = ax.plot(xs1, Time_DS1 * 0, 'k--', dashes=(10, 5), lw=0.5)
    artists.append(a[0])
    a = ax.plot(xs1, qplot1, label='\(q\) (Sv)')
    artists.append(a[0])
    a = ax.plot(xs1, DeltaS1[0, :], label='\(\Delta S\)')
    artists.append(a[0])

    ax.set_xlabel('Time (kyr)')
    ax.set_title('(e) THC transport \(q\)')
    ax.legend()

    return fig, ax, artists


def main():
    S0 = 35.0
    T0 = 24.0
    S1, S2 = 12, 20
    T1, T2 = 1, 10
    area = (50000.0e3) ** 2
    depth = 4000
    rho0 = 1
    alpha = 0.2 # (kg/m^3) K^-1
    beta = 0.8 # (kg/m^3) ppt^-1
    k = 10e9 # (m^3/s)/(kg/m^3) # 3e-8; # sec^-1

    time = 0
    time_max = 50000 * YEAR
    DeltaS_range = np.arange(-3, 0, 0.01)
    model = BoxModel(S0, S1, S2, T0, T1, T2, alpha, beta, k, area, depth)

    fig, ax = plt.subplots()
    _ = plot_time_series(model, time_max, fig=fig, ax=ax)
    # _ = plot_rhs(model, time, time_max, DeltaS_range, fig=fig, ax=ax)
    plt.show()
    exit()
    n_frames = 30
    sin_range = np.sin(np.linspace(0, 10, num=n_frames))

    def animate(i):
        ax.clear()
        # ax.set_autoscale_on(False)
        # ax.set_ylim([-4, 15])
        # ax.set_xlim([0, 50])
        _, _, artists = plot_time_series(model, time_max, fig=fig, ax=ax)
        # model.T2 += 0.1
        model.T2 = sin_range[i]
        ax.set_title(f'\(T_1 = {model.T1}\), \(T_2 = {model.T2}\)')
        return artists
        
    anim = FuncAnimation(
        fig, animate, interval=1, blit=True, repeat=True, frames=n_frames,
    )
    anim.save('box.gif', dpi=300, writer=PillowWriter(fps=5))

    # plt.show()

    exit()
    Fs_range = np.arange(0, 5, 0.01)
    Fs_range = scale_Fs(Fs_range, area, S0, year)
    model = BoxModel(S0, S1, S2, T1, T2, alpha, beta, k, area, depth)
    DeltaS_steady, q_steady = model.simulate(Fs_range)


if __name__ == '__main__':
    main()
