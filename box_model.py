from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import combine_forcing

# Constants
YEAR = 365 * 24 * 3600
Sv = 1.e9 # m^3/sec


class BoxModel:
    """
    Implementation of Stommel's box model.
    """
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

    def Fs_constant(self, time, time_max, flux=2, period=20, FW_min=-0.1, FW_max=5):
        return flux * self.area * self.S0 / YEAR

    def Fs_linear(self, time, time_max, period=20, FW_min=-0.1, FW_max=5):
        # Linear interpolation between minimum F and maximum F
        flux = FW_min + (FW_max - FW_min) * time / time_max
        return flux * self.area * self.S0 / YEAR
    
    def Fs_sinusoidal(self, time, time_max, period=20, FW_min=-5, FW_max=5):
        # Sinusoidal interpolation between minimum F and maximum F
        half_range = (FW_max - FW_min) / 2
        amplitude_mult = time / time_max
        flux = FW_min + half_range + np.sin(period * time / time_max) * half_range
        ret = amplitude_mult * flux * self.area * self.S0 / YEAR
        return ret
    
    def Fs_meta(self, time, time_max, ls_F_type, ls_F_length, combination=None):
        if combination is None:
            combination = combine_forcing(0, time_max, ls_F_type, ls_F_length)
        return combination[int(time)]

    def rhs_S(self, time, time_max, fn_forcing=None, forcing_kwargs=dict(), DeltaT=None, DeltaS=None):
        """
        Implements the right-hand side of the derivative of S wrt. t (time).
        """
        if fn_forcing is None:
            fn_forcing = self.Fs_sinusoidal

        DeltaT = self.DeltaT if DeltaT is None else DeltaT
        DeltaS = self.DeltaS if DeltaS is None else DeltaS
        
        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        # Optional: with S0
        # rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaS + 2 * F_s * self.S0)
        rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaS + 2 * F_s)

        return rhs / self.V
    
    def rhs_T(self, time, time_max, fn_forcing=None, forcing_kwargs=dict(), DeltaT=None, DeltaS=None):
        """
        Implements the right-hand side of the derivative of T wrt. t (time).
        """
        if fn_forcing is None:
            fn_forcing = self.Fs_sinusoidal

        DeltaT = self.DeltaT if DeltaT is None else DeltaT
        DeltaS = self.DeltaS if DeltaS is None else DeltaS
        
        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        # Optional: with S0
        # rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaS + 2 * F_s * self.S0)
        rhs = -(2 * abs(self.q(DeltaT, DeltaS)) * DeltaT + 2 * F_s)

        return rhs / self.V

    def simulate(self, Fs_range):
        """
        Computes the steady state solutions as well as the q values
        for the above values of F.
        """
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
    Fs_range_normalized = np.copy(Fs_range)
    if normalize:
        Fs_range_normalized /= Fs_to_m_per_year(S0, area, YEAR)
    
    fig, ax = plt.subplots(ncols=2)

    # Plot all three solutions for Delta S as function of Fs in units of m/year:
    ax[0].plot(Fs_range_normalized, DeltaS_steady[0, :], 'r.', markersize=1)
    ax[0].plot(Fs_range_normalized, DeltaS_steady[1, :], 'g.', markersize=1)
    ax[0].plot(Fs_range_normalized, DeltaS_steady[2, :], 'b.', markersize=1)
    
    # Plot a dash think line marking the zero value:
    ax[0].plot(Fs_range_normalized, np.zeros(DeltaS_steady.shape[1]), 'k--', dashes=(10, 5), linewidth=0.5)
    ax[0].title('(a) steady states')
    ax[0].xlabel('$F_s$ (m/year)');
    ax[0].ylabel('$\Delta S$');
    ax[0].xlim([min(Fs_range_normalized), max(Fs_range_normalized)])
    
    # Plot all three solutions for q (in Sv) as function of Fs in units of m/year:
    ax[1].plot(Fs_range_normalized, q_steady[0, :], 'r.', markersize=1)
    ax[1].plot(Fs_range_normalized, q_steady[1, :], 'g.', markersize=1)
    ax[1].plot(Fs_range_normalized, q_steady[2, :], 'b.', markersize=1)
    ax[1].plot(Fs_range_normalized, np.zeros(q_steady.shape[1]), 'k--', dashes=(10, 5), linewidth=0.5)
    ax[1].title('(b) steady states')
    ax[1].xlabel('$F_s$ (m/year)')
    ax[1].ylabel('$q$ (Sv)')

    fig.tight_layout()

    return fig, ax


def plot_rhs(model, time, time_max, DeltaS_range, fig=None, ax=None):
    """
    Usage:
    fig, ax = None, None
    time, time_max = 0, 150000 * box_model.YEAR
    DeltaS_range = np.arange(-3, 0, 0.01)
    _ = plot_rhs(model, time, time_max, DeltaS_range, fig=fig, ax=ax)
    """
    DeltaS_range = np.arange(-3, 0, 0.01)
    rhs = np.zeros(len(DeltaS_range))

    for i in range(len(DeltaS_range)):
        DeltaS = DeltaS_range[i]
        rhs[i] = model.rhs_S(time, time_max, DeltaS=DeltaS)

    rhs /= np.std(rhs)

    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    
    ax.plot(DeltaS_range, rhs, 'k-', lw=2)
    ax.plot(DeltaS_range, rhs * 0, 'k--', dashes=(10, 5), lw=0.5)

    # Superimpose color markers of the 3 solutions
    Fs = model.Fs_constant(2)
    yy = model.steady_states(Fs, model.alpha * model.DeltaT) / model.beta

    ax.plot(yy[0], 0, 'ro', markersize=10)
    ax.plot(yy[1], 0, 'go', markersize=10)
    ax.plot(yy[2], 0, 'bo', markersize=10, fillstyle='none')
    ax.set_title('(c) stability')
    ax.set_xlabel('\(\Delta S\)')
    ax.set_ylabel('\(d\Delta S/dt\)')

    return fig, ax


def get_time_series(
        model, time_max, forcing='sinusoidal', forcing_kwargs=dict(), fig=None, ax=None,
    ):
    teval = np.arange(0, time_max, time_max / 1000)
    tspan = (teval[0], teval[-1])
    
    if forcing == 'linear':
        fn_forcing = model.Fs_linear
    elif forcing == 'sinusoidal':
        fn_forcing = model.Fs_sinusoidal
    elif forcing == 'constant':
        fn_forcing = model.Fs_constant
    elif forcing == 'meta':
        ls_F_type = ['s']
        ls_F_length = [time_max]
        combination = combine_forcing(0, time_max, ls_F_type, ls_F_length)
        fn_forcing = partial(
            model.Fs_meta,
            ls_F_type=ls_F_type,
            ls_F_length=ls_F_length,
            combination=combination,
        )

    sol_DS = sp.integrate.solve_ivp(
        fun=lambda time, DeltaS: model.rhs_S(
            time,
            time_max,
            fn_forcing=fn_forcing,
            forcing_kwargs=forcing_kwargs,
            DeltaS=DeltaS,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )

    sol_DT = sp.integrate.solve_ivp(
        fun=lambda time, DeltaT: model.rhs_T(
            time,
            time_max,
            fn_forcing=fn_forcing,
            forcing_kwargs=forcing_kwargs,
            DeltaT=DeltaT,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )

    Time_DS, Time_DT = sol_DS.t, sol_DT.t
    DeltaS, DeltaT = sol_DS.y[0, :], sol_DT.y[0, :]

    FWplot = np.zeros(len(Time_DS))
    qplot = np.zeros(len(Time_DS))

    for i, t in enumerate(Time_DS):
        FWplot[i] = fn_forcing(t, time_max, **forcing_kwargs)
        # qplot[i] = model.q(model.DeltaT, DeltaS[0, i])
        qplot[i] = model.q(DeltaT[i], DeltaS[i])

    xs_S = Time_DS / YEAR / 1000
    xs_T = Time_DT / YEAR / 1000

    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(ncols=4, figsize=(9, 7))

    F = FWplot / Fs_to_m_per_year(model.S0, model.area)
    ax[0].plot(xs_S, F, 'b-', markersize=1)
    ax[0].plot(xs_S, Time_DS*0, 'k--', dashes=(10, 5), linewidth=0.5)
    ax[0].set_xlabel('Time (kyr)')
    ax[0].set_ylabel('$F$ (m/yr)')
    ax[0].set_title('Forcing')

    y = qplot / Sv
    ax[1].plot(xs_S, y, 'b-', markersize=1)
    ax[1].plot(xs_S, Time_DS*0, 'k--', dashes=(10, 5), lw=0.5)
    ax[1].set_xlabel('Time (kyr)')
    ax[1].set_ylabel('$q$ (Sv)')
    ax[1].set_title('THC transport')

    ax[2].plot(xs_S, DeltaS, 'b-', markersize=1)
    ax[2].plot(xs_S, Time_DS*0, 'k--', dashes=(10, 5), lw=0.5)
    ax[2].set_title('$\Delta S$ vs time')
    ax[2].set_xlabel('Time (kyr)')
    ax[2].set_ylabel('$\Delta S$')

    ax[3].plot(xs_T, DeltaT, 'b-', markersize=1)
    ax[3].plot(xs_T, Time_DT*0, 'k--', dashes=(10, 5), lw=0.5)
    ax[3].set_title('$\Delta T$ vs time')
    ax[3].set_xlabel('Time (kyr)')
    ax[3].set_ylabel('$\Delta T$')

    fig.tight_layout()

    return fig, ax, F, DeltaS, DeltaT, y



def animation(filename, fig, ax, model, time_max):
    n_frames = 30
    sin_range = np.sin(np.linspace(0, 10, num=n_frames))

    def animate(i):
        ax.clear()
        _, _, artists = get_time_series(model, time_max, fig=fig, ax=ax)
        # model.T2 += 0.1
        model.T2 = sin_range[i]
        ax.set_title(f'\(T_1 = {model.T1}\), \(T_2 = {model.T2}\)')
        return artists
    
    anim = FuncAnimation(
        fig, animate, interval=1, blit=True, repeat=True, frames=n_frames,
    )
    anim.save(filename, dpi=300, writer=PillowWriter(fps=5))


def simple_simulation(area, year, S0, S1, S2, T1, T2, alpha, beta, k, depth):
    Fs_range = np.arange(0, 5, 0.01)
    Fs_range = scale_Fs(Fs_range, area, S0, year)
    model = BoxModel(S0, S1, S2, T1, T2, alpha, beta, k, area, depth)
    return model.simulate(Fs_range)
