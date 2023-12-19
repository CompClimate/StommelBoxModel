"""Implements things related to the Stommel Box Model."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seawater as sw
from matplotlib.animation import FuncAnimation, PillowWriter

from .forcing import Forcing

# Constants
YEAR = 365 * 24 * 3600
Sv = 1.0e9  # m^3/sec


class BoxModel:
    """Implementation of Stommel's box model."""

    def __init__(self, S0, S1, S2, T0, T1, T2, alpha, beta, k, area, depth, years):
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
        self.years = years * YEAR

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
        """Implements the AMOC volume transport q with a linear density approximation."""
        # THC transport in m^3/sec as function of temperature and salinity difference between the boxes

        rho_1_2_diff = self.alpha * DeltaT - self.beta * DeltaS
        flow = self.k * rho_1_2_diff

        return flow

    def q80(self, T1, T2, S1, S2):
        """Implements the AMOC volume transport q with a nonlinear density approximation using EOS80."""
        return self.k * (sw.eos80.dens(S1, T1, 0) - sw.eos80.dens(S2, T2, 0))

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

    def F_constant(self, time, time_max, flux=2, period=20, FW_min=-0.1, FW_max=5):
        return flux * self.area * self.S0 / YEAR

    def F_linear(self, time, time_max, period=20, FW_min=-0.1, FW_max=5):
        # Linear interpolation between minimum F and maximum F
        flux = FW_min + (FW_max - FW_min) * time / time_max
        return flux * self.area * self.S0 / YEAR

    def F_sinusoidal(
        self, time, time_max, period=20, FW_min=-5, FW_max=5, nonstationary=False
    ):
        # Sinusoidal interpolation between minimum F and maximum F
        half_range = (FW_max - FW_min) / 2
        amplitude_mult = time / time_max if nonstationary else 1
        sin_arg = period * time / time_max
        flux = FW_min + half_range + np.sin(sin_arg) * half_range
        ret = amplitude_mult * flux * self.area * self.S0 / YEAR
        return ret

    def rhs_S(
        self,
        time,
        time_max,
        fn_forcing=None,
        forcing_kwargs=dict(),
        DeltaT=None,
        DeltaS=None,
    ):
        """Implements the right-hand side of the derivative of Delta S wrt t (time)."""
        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        DeltaT = self.DeltaT if DeltaT is None else DeltaT
        DeltaS = self.DeltaS if DeltaS is None else DeltaS

        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q(DeltaT, DeltaS)
        rhs = -(2 * abs(q) * DeltaS + 2 * F_s)

        return rhs / self.V

    def rhs_T(
        self,
        time,
        time_max,
        fn_forcing=None,
        forcing_kwargs=dict(),
        DeltaT=None,
        DeltaS=None,
    ):
        """Implements the right-hand side of the derivative of Delta T wrt t (time)."""
        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        DeltaT = self.DeltaT if DeltaT is None else DeltaT
        DeltaS = self.DeltaS if DeltaS is None else DeltaS

        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q(DeltaT, DeltaS)
        rhs = -(2 * abs(q) * DeltaT + 2 * F_s)

        return rhs / self.V

    def rhs_S1(
        self,
        time,
        time_max,
        S1,
        S2,
        T1,
        T2,
        fn_forcing=None,
        forcing_kwargs=dict(),
    ):
        """Implements the right-hand side of the derivative of S_1 wrt t (time)."""
        V_1 = self.V / 2

        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q80(T1, T2, S1, S2)
        return (abs(q) * (S2 - S1) - F_s) / V_1

    def rhs_S2(
        self,
        time,
        time_max,
        S1,
        S2,
        T1,
        T2,
        fn_forcing=None,
        forcing_kwargs=dict(),
    ):
        """Implements the right-hand side of the derivative of S_2 wrt t (time)."""
        V_2 = self.V / 2

        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        F_s = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q80(T1, T2, S1, S2)
        return (abs(q) * (S1 - S2) + F_s) / V_2

    def rhs_T1(
        self,
        time,
        time_max,
        S1,
        S2,
        T1,
        T2,
        fn_forcing=None,
        forcing_kwargs=dict(),
    ):
        """Implements the right-hand side of the derivative of T_1 wrt t (time)."""
        V_1 = self.V / 2

        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        F_t = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q80(T1, T2, S1, S2)
        return (abs(q) * (T2 - T1) - F_t) / V_1

    def rhs_T2(
        self,
        time,
        time_max,
        S1,
        S2,
        T1,
        T2,
        fn_forcing=None,
        forcing_kwargs=dict(),
    ):
        """Implements the right-hand side of the derivative of T_2 wrt t (time)."""
        V_1 = self.V / 2

        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        F_t = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q80(T1, T2, S1, S2)
        return (abs(q) * (T1 - T2) + F_t) / V_1

    def simulate(self, Fs_range):
        """Computes the steady state solutions as well as the q values for the above values of
        F."""
        DeltaS_steady = np.zeros((3, len(Fs_range)))
        q_steady = np.zeros((3, len(Fs_range)))

        for i, Fs in enumerate(Fs_range):
            Y = self.steady_states(Fs, self.alpha * self.DeltaT)
            # translate Y solution to a solution for DeltaS:
            DeltaS_steady[:, i] = Y / self.beta
            for j in range(3):
                q_steady[j, i] = self.q(self.DeltaT, DeltaS_steady[j, i])

        return DeltaS_steady, q_steady

    def forcing_from_str(self, forcing_name):
        if forcing_name == "linear":
            fn_forcing = self.F_linear
        elif forcing_name == "sinusoidal":
            fn_forcing = self.F_sinusoidal
        elif forcing_name == "constant":
            fn_forcing = self.F_constant

        return fn_forcing


def scale_Fs(Fs, area, S0, year):
    return Fs * area * S0 / year


def Fs_to_m_per_year(S0, area):
    return S0 * area / YEAR


def Ft_to_m_per_year(T0, area):
    return T0 * area / YEAR


def plot_steady(Fs_range, DeltaS_steady, q_steady, S0, area, normalize=True):
    Fs_range_normalized = np.copy(Fs_range)
    if normalize:
        Fs_range_normalized /= Fs_to_m_per_year(S0, area, YEAR)

    fig, ax = plt.subplots(ncols=2)

    # Plot all three solutions for Delta S as function of Fs in units of m/year:
    ax[0].plot(Fs_range_normalized, DeltaS_steady[0, :], "r.", markersize=1)
    ax[0].plot(Fs_range_normalized, DeltaS_steady[1, :], "g.", markersize=1)
    ax[0].plot(Fs_range_normalized, DeltaS_steady[2, :], "b.", markersize=1)

    # Plot a dash think line marking the zero value:
    ax[0].plot(
        Fs_range_normalized,
        np.zeros(DeltaS_steady.shape[1]),
        "k--",
        dashes=(10, 5),
        linewidth=0.5,
    )
    ax[0].title("(a) steady states")
    ax[0].xlabel("$F_s$ (m/year)")
    ax[0].ylabel(r"$\Delta S$")
    ax[0].xlim([min(Fs_range_normalized), max(Fs_range_normalized)])

    # Plot all three solutions for q (in Sv) as function of Fs in units of m/year:
    ax[1].plot(Fs_range_normalized, q_steady[0, :], "r.", markersize=1)
    ax[1].plot(Fs_range_normalized, q_steady[1, :], "g.", markersize=1)
    ax[1].plot(Fs_range_normalized, q_steady[2, :], "b.", markersize=1)
    ax[1].plot(
        Fs_range_normalized,
        np.zeros(q_steady.shape[1]),
        "k--",
        dashes=(10, 5),
        linewidth=0.5,
    )
    ax[1].title("(b) steady states")
    ax[1].xlabel("$F_s$ (m/year)")
    ax[1].ylabel("$q$ (Sv)")

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

    ax.plot(DeltaS_range, rhs, "k-", lw=2)
    ax.plot(DeltaS_range, rhs * 0, "k--", dashes=(10, 5), lw=0.5)

    # Superimpose color markers of the 3 solutions
    Fs = model.Fs_constant(2)
    yy = model.steady_states(Fs, model.alpha * model.DeltaT) / model.beta

    ax.plot(yy[0], 0, "ro", markersize=10)
    ax.plot(yy[1], 0, "go", markersize=10)
    ax.plot(yy[2], 0, "bo", markersize=10, fillstyle="none")
    ax.set_title("(c) stability")
    ax.set_xlabel(r"\(\Delta S\)")
    ax.set_ylabel(r"\(d\Delta S/dt\)")

    return fig, ax


def get_time_series_linear_(
    model: BoxModel,
    s_forcing: Forcing,
    t_forcing: Forcing,
):
    teval = np.arange(0, model.years, model.years / 1000)
    tspan = (teval[0], teval[-1])

    Fs_forcing = model.forcing_from_str(s_forcing.forcing)
    Ft_forcing = model.forcing_from_str(t_forcing.forcing)

    sol_DS = sp.integrate.solve_ivp(
        fun=lambda time, y: model.rhs_S(
            time,
            model.years,
            DeltaS=y,
            fn_forcing=Fs_forcing,
            forcing_kwargs=s_forcing.forcing_kwargs,
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
            model.years,
            fn_forcing=Ft_forcing,
            forcing_kwargs=t_forcing.forcing_kwargs,
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

    Fs_plot = np.zeros(len(Time_DS))
    qplot = np.zeros(len(Time_DS))
    for i, t in enumerate(Time_DS):
        Fs_plot[i] = Fs_forcing(t, model.years, **s_forcing.forcing_kwargs)
        qplot[i] = model.q(DeltaT[i], DeltaS[i])

    F_s = Fs_plot / Fs_to_m_per_year(model.S0, model.area)
    q = qplot / Sv

    series_dict = {
        "times": [Time_DS, Time_DT],
        "features": {
            "variables": {"DeltaS": DeltaS, "DeltaT": DeltaT},
            "forcings": {"Fs": F_s},
        },
        "q": q,
        "units": {
            "q": "Sv",
            "DeltaS": "ppt",
            "DeltaT": r"\(\tccentigrade\)",
            "Fs": "m / yr",
        },
        "latex": {
            "variables": {"DeltaS": r"\(\Delta S\)", "DeltaT": r"\(\Delta T\)"},
            "forcings": {"Fs": r"\(F_s\)"},
        },
    }

    return series_dict


def get_time_series_nonlinear_(
    model: BoxModel,
    s_forcing: Forcing,
    t_forcing: Forcing,
):
    teval = np.arange(0, model.years, model.years / 1000)
    tspan = (teval[0], teval[-1])

    Fs_forcing = model.forcing_from_str(s_forcing.forcing)
    Ft_forcing = model.forcing_from_str(t_forcing.forcing)

    sol_S1 = sp.integrate.solve_ivp(
        fun=lambda time, y: model.rhs_S1(
            time,
            model.years,
            S1=y,
            S2=model.S2,
            T1=model.T1,
            T2=model.T2,
            fn_forcing=Fs_forcing,
            forcing_kwargs=s_forcing.forcing_kwargs,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )
    sol_S2 = sp.integrate.solve_ivp(
        fun=lambda time, y: model.rhs_S2(
            time,
            model.years,
            S1=model.S1,
            S2=y,
            T1=model.T1,
            T2=model.T2,
            fn_forcing=Fs_forcing,
            forcing_kwargs=s_forcing.forcing_kwargs,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )

    sol_T1 = sp.integrate.solve_ivp(
        fun=lambda time, y: model.rhs_T1(
            time,
            model.years,
            S1=model.S1,
            S2=model.S2,
            T1=y,
            T2=model.T2,
            fn_forcing=Ft_forcing,
            forcing_kwargs=t_forcing.forcing_kwargs,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )
    sol_T2 = sp.integrate.solve_ivp(
        fun=lambda time, y: model.rhs_T2(
            time,
            model.years,
            S1=model.S1,
            S2=model.S2,
            T1=model.T1,
            T2=y,
            fn_forcing=Ft_forcing,
            forcing_kwargs=t_forcing.forcing_kwargs,
        ),
        vectorized=False,
        y0=[0],
        t_span=tspan,
        t_eval=teval,
        dense_output=True,
    )

    Time_S1, Time_S2, Time_T1, Time_T2 = sol_S1.t, sol_S2.t, sol_T1.t, sol_T2.t
    S1, S2, T1, T2 = sol_S1.y[0, :], sol_S2.y[0, :], sol_T1.y[0, :], sol_T2.y[0, :]

    Fs_plot = np.zeros(len(Time_S1))
    Ft_plot = np.zeros(len(Time_T1))
    qplot = np.zeros(len(Time_S1))

    for i, t in enumerate(Time_S1):
        Fs_plot[i] = Fs_forcing(t, model.years, **s_forcing.forcing_kwargs)
        Ft_plot[i] = Ft_forcing(t, model.years, **t_forcing.forcing_kwargs)
        qplot[i] = model.q80(T1[i], T2[i], S1[i], S2[i])

    F_s = Fs_plot / Fs_to_m_per_year(model.S0, model.area)
    F_t = Ft_plot / Ft_to_m_per_year(model.T0, model.area)
    q = qplot / Sv

    series_dict = {
        "times": [Time_S1, Time_S2, Time_T1, Time_T2],
        "features": {
            "variables": {"S1": S1, "S2": S2, "T1": T1, "T2": T2},
            "forcings": {"Fs": F_s, "Ft": F_t},
        },
        "q": q,
        "units": {
            "q": "Sv",
            "S1": "ppt",
            "S2": "ppt",
            "T1": r"\(\tccentigrade\)",
            "T2": r"\(\tccentigrade\)",
            "Fs": "m / year",
            "Ft": r"\(\tccentigrade\) / year",
        },
        "latex": {
            "variables": {
                "S1": r"\(S_1\)",
                "S2": r"\(S_2\)",
                "T1": r"\(T_1\)",
                "T2": r"\(T_2\)",
            },
            "forcings": {"Fs": r"\(F_s\)", "Ft": r"\(F_t\)"},
        },
    }

    return series_dict


def get_time_series(
    model: BoxModel,
    s_forcing: Forcing,
    t_forcing: Forcing,
    nonlinear_density: bool,
):
    data = (
        get_time_series_nonlinear_(model, s_forcing, t_forcing)
        if nonlinear_density
        else get_time_series_linear_(model, s_forcing, t_forcing)
    )
    return data


def plot_time_series(series_dict):
    variables = series_dict["features"]["variables"]
    forcings = series_dict["features"]["forcings"]
    latex_variables = series_dict["latex"]["variables"]
    latex_forcings = series_dict["latex"]["forcings"]
    units = series_dict["units"]

    fig, ax = plt.subplots(ncols=3, figsize=(12, 8))

    time = series_dict["times"][0]
    x_label = "Time (kyr)"

    xs_S = time / YEAR / 1000
    q = series_dict["q"]

    # Plot forcings
    ax_F = ax[0]
    ax_F.set_xlabel(x_label)
    for F_name, F in forcings.items():
        ax_F.plot(
            xs_S,
            F,
            label=f"{latex_forcings[F_name]} ({units[F_name]})",
        )
        ax_F.plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    ax[0].set_box_aspect(1)
    ax[0].legend()

    # Plot the input variables
    ax[1].set_xlabel(x_label)
    # ax[1].set_ylabel("ppt")
    for var_name, value in variables.items():
        ax[1].plot(
            xs_S,
            value,
            label=f"{latex_variables[var_name]} ({units[var_name]})",
        )
        ax[1].plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    ax[1].set_box_aspect(1)
    ax[1].legend()

    # Plot q
    ax[2]
    ax[2].plot(xs_S, q)
    ax[2].plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel(rf"\(q\) ({units['q']})")
    ax[2].set_box_aspect(1)

    fig.tight_layout()

    return fig


def animation(filename, fig, ax, model):
    n_frames = 30
    sin_range = np.sin(np.linspace(0, 10, num=n_frames))

    def animate(i):
        ax.clear()
        _, _, artists = get_time_series(model, model.years, fig=fig, ax=ax)
        model.T2 = sin_range[i]
        ax.set_title(rf"\(T_1 = {model.T1}\), \(T_2 = {model.T2}\)")
        return artists

    anim = FuncAnimation(
        fig,
        animate,
        interval=1,
        blit=True,
        repeat=True,
        frames=n_frames,
    )
    anim.save(filename, dpi=300, writer=PillowWriter(fps=5))


def simple_simulation(area, year, S0, S1, S2, T1, T2, alpha, beta, k, depth):
    Fs_range = np.arange(0, 5, 0.01)
    Fs_range = scale_Fs(Fs_range, area, S0, year)
    model = BoxModel(S0, S1, S2, T1, T2, alpha, beta, k, area, depth)
    return model.simulate(Fs_range)
