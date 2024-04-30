"""Implements things related to the Stommel Box Model."""

import os.path as osp
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seawater as sw
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig
from scipy.integrate import odeint, solve_ivp

from .forcing import Forcing

YEAR = 365 * 24 * 3600
Sv = 1.0e9


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
        # THC transport in m^3/sec as function of temperature and salinity
        # difference between the boxes

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

        F_t = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q(DeltaT, DeltaS)
        rhs = -(2 * abs(q) * DeltaT + 2 * F_t)

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
        # V_1 = self.V / 2
        V_1 = self.V

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
        # V_2 = self.V / 2
        V_2 = self.V

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
        # V_1 = self.V / 2
        V_1 = self.V

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
        # V_2 = self.V / 2
        V_2 = self.V

        if fn_forcing is None:
            fn_forcing = self.F_sinusoidal

        F_t = fn_forcing(time, time_max, **forcing_kwargs)
        q = self.q80(T1, T2, S1, S2)
        return (abs(q) * (T1 - T2) + F_t) / V_2

    def simulate(self, Fs_range):
        """
        Computes the steady state solutions as well as the q values for the above values of F.
        """
        DeltaS_steady = np.zeros((3, len(Fs_range)))
        q_steady = np.zeros((3, len(Fs_range)))

        for i, Fs in enumerate(Fs_range):
            Y = self.steady_states(Fs, self.alpha * self.DeltaT)
            # Translate Y solution to a solution for DeltaS:
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
        Fs_range_normalized /= Fs_to_m_per_year(S0, area)

    fig, ax = plt.subplots(ncols=2)

    # Plot all three solutions for Delta S as function of Fs in units of
    # m/year:
    ax[0].plot(Fs_range_normalized, DeltaS_steady[0, :], "r.")
    ax[0].plot(Fs_range_normalized, DeltaS_steady[1, :], "g.")
    ax[0].plot(Fs_range_normalized, DeltaS_steady[2, :], "b.")

    # Plot a dash think line marking the zero value:
    ax[0].plot(
        Fs_range_normalized,
        np.zeros(DeltaS_steady.shape[1]),
        "k--",
        dashes=(10, 5),
        linewidth=0.5,
    )
    ax[0].set_title("(a) steady states")
    ax[0].set_xlabel(r"\(F_s\) (m/year)")
    ax[0].set_ylabel(r"\(\Delta S\)")
    # ax[0].set_xlim([min(Fs_range_normalized), max(Fs_range_normalized)])

    # Plot all three solutions for q (in Sv) as function of Fs in units of
    # m/year:
    ax[1].plot(Fs_range_normalized, q_steady[0, :], "r.")
    ax[1].plot(Fs_range_normalized, q_steady[1, :], "g.")
    ax[1].plot(Fs_range_normalized, q_steady[2, :], "b.")
    ax[1].plot(
        Fs_range_normalized,
        np.zeros(q_steady.shape[1]),
        "k--",
        dashes=(10, 5),
        linewidth=0.5,
    )
    ax[1].set_title("(b) steady states")
    ax[1].set_xlabel(r"\(F_s\) (m/year)")
    ax[1].set_ylabel(r"\(q\) (Sv)")

    fig.tight_layout()

    return fig, ax


def plot_rhs(
    model, time, time_max, DeltaS_range, forcing_name="linear", fig=None, ax=None
):
    """
    Usage:
    fig, ax = None, None
    time, time_max = 0, 150000 * box_model.YEAR
    DeltaS_range = np.arange(-3, 0, 0.01)
    _ = plot_rhs(model, time, time_max, DeltaS_range, fig=fig, ax=ax)
    """
    rhs = np.zeros(len(DeltaS_range))

    for i in range(len(DeltaS_range)):
        DeltaS = DeltaS_range[i]
        rhs[i] = model.rhs_S(time, time_max, DeltaS=DeltaS)

    rhs /= np.std(rhs)

    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    ax.plot(DeltaS_range, rhs, "k-", lw=2)
    ax.plot(DeltaS_range, rhs * 0, "k--", dashes=(10, 5), lw=0.5)

    # Superimpose color markers of the 3 solutions
    Fs = model.forcing_from_str(forcing_name)(time, time_max)
    yy = model.steady_states(Fs, model.alpha * model.DeltaT) / model.beta

    ax.plot(yy[0], 0, "ro", markersize=10)
    ax.plot(yy[1], 0, "go", markersize=10)
    ax.plot(yy[2], 0, "bo", markersize=10, fillstyle="none")
    ax.set_title("(c) stability")
    ax.set_xlabel(r"\(\Delta S\)")
    ax.set_ylabel(r"\(d\Delta S/dt\)")

    fig.tight_layout()

    return fig, ax


def get_time_series_linear_(
    model: BoxModel,
    s_forcing: Forcing,
    solution_cfg: DictConfig,
    init_time_series_cfg: DictConfig,
    t_forcing: Optional[Forcing] = None,
):
    teval = np.arange(0, model.years, model.years / 1000)
    tspan = (teval[0], teval[-1])

    Fs_forcing = model.forcing_from_str(s_forcing.forcing)
    if t_forcing is not None:
        Ft_forcing = model.forcing_from_str(t_forcing.forcing)

    y0_S = (
        model.DeltaS
        if init_time_series_cfg.get("DeltaS_init") is None
        else init_time_series_cfg.DeltaS_init
    )
    if t_forcing is not None:
        y0_T = (
            model.DeltaT
            if init_time_series_cfg.get("DeltaT_init") is None
            else init_time_series_cfg.DeltaT_init
        )

    if solution_cfg["integration_method"] == "solve_ivp":
        common_kwargs = dict(
            t_span=tspan, t_eval=teval, **solution_cfg.get("kwargs", {})
        )
        sol_DS = solve_ivp(
            fun=lambda t, y: model.rhs_S(
                time=t,
                time_max=model.years,
                DeltaS=y,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=[y0_S],
            **common_kwargs,
        )
        if t_forcing is not None:
            sol_DT = solve_ivp(
                fun=lambda t, y: model.rhs_T(
                    time=t,
                    time_max=model.years,
                    DeltaT=y,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=[y0_T],
                **common_kwargs,
            )
            Time_DT = sol_DT.t
            DeltaT = sol_DT.y[0, :]

        Time_DS = sol_DS.t
        DeltaS = sol_DS.y[0, :]
    elif solution_cfg["integration_method"] == "odeint":
        common_kwargs = dict(t=teval, **solution_cfg.get("kwargs", {}))
        sol_DS = odeint(
            func=lambda y, t: model.rhs_S(
                time=t,
                time_max=model.years,
                DeltaS=y,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=y0_S,
            **common_kwargs,
        )
        if t_forcing is not None:
            sol_DT = odeint(
                func=lambda y, t: model.rhs_T(
                    time=t,
                    time_max=model.years,
                    DeltaT=y,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=y0_T,
                **common_kwargs,
            )
            Time_DT = teval
            DeltaT = sol_DT

        Time_DS = teval
        DeltaS = sol_DS

    Fs_plot = np.zeros(len(Time_DS))
    if t_forcing is not None:
        Ft_plot = np.zeros(len(Time_DT))
    qplot = np.zeros(len(Time_DS))
    for i, t in enumerate(Time_DS):
        Fs_plot[i] = Fs_forcing(t, model.years, **s_forcing.forcing_kwargs)
        if t_forcing is not None:
            Ft_plot[i] = Ft_forcing(t, model.years, **t_forcing.forcing_kwargs)
            qplot[i] = model.q(DeltaT[i], DeltaS[i])
        else:
            qplot[i] = model.q(model.DeltaT, DeltaS[i])

    F_s = Fs_plot / Fs_to_m_per_year(model.S0, model.area)
    if t_forcing is not None:
        F_t = Ft_plot / Ft_to_m_per_year(model.T0, model.area)
    q = qplot / Sv

    series_dict = {
        "times": [Time_DS],
        "features": {
            "variables": {"DeltaS": DeltaS},
            "forcings": {"Fs": F_s},
        },
        "q": q,
        "units": {
            "q": "Sv",
            "DeltaS": "ppt",
            "Fs": "m / yr",
        },
        "latex": {
            "variables": {"DeltaS": r"\(\Delta S\)"},
            "forcings": {"Fs": r"\(F_s\)"},
        },
    }

    if t_forcing is not None:
        series_dict["times"] += Time_DT
        series_dict["features"]["variables"].update({"DeltaT": DeltaT})
        series_dict["features"]["forcings"].update({"Ft": F_t})
        series_dict["units"].update(
            {"DeltaT": r"\(\tccentigrade\)", "Ft": r"\(\tccentigrade\) / yr"}
        )
        series_dict["latex"]["variables"].update({"DeltaT": r"\(\Delta T\)"})
        series_dict["latex"]["forcings"].update({"Ft": r"\(F_t\)"})

    return series_dict


def get_time_series_nonlinear_(
    model: BoxModel,
    s_forcing: Forcing,
    solution_cfg: DictConfig,
    init_time_series_cfg: DictConfig,
    t_forcing: Optional[Forcing] = None,
):
    teval = np.arange(0, model.years, model.years / 1000)
    tspan = (teval[0], teval[-1])

    Fs_forcing = model.forcing_from_str(s_forcing.forcing)
    y0_S1 = (
        model.S1
        if init_time_series_cfg.S1_init is None
        else init_time_series_cfg.S1_init
    )
    y0_S2 = (
        model.S2
        if init_time_series_cfg.S2_init is None
        else init_time_series_cfg.S2_init
    )

    if t_forcing is not None:
        Ft_forcing = model.forcing_from_str(t_forcing.forcing)
        y0_T1 = (
            model.T1
            if init_time_series_cfg.T1_init is None
            else init_time_series_cfg.T1_init
        )
        y0_T2 = (
            model.T2
            if init_time_series_cfg.T2_init is None
            else init_time_series_cfg.T2_init
        )

    if solution_cfg["integration_method"] == "solve_ivp":
        common_kwargs = dict(
            t_span=tspan,
            t_eval=teval,
            **solution_cfg.get("kwargs", {}),
        )
        sol_S1 = solve_ivp(
            fun=lambda t, y: model.rhs_S1(
                time=t,
                time_max=model.years,
                S1=y,
                S2=model.S2,
                T1=model.T1,
                T2=model.T2,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=[y0_S1],
            **common_kwargs,
        )
        sol_S2 = solve_ivp(
            fun=lambda t, y: model.rhs_S2(
                time=t,
                time_max=model.years,
                S1=model.S1,
                S2=y,
                T1=model.T1,
                T2=model.T2,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=[y0_S2],
            **common_kwargs,
        )
        if t_forcing is not None:
            sol_T1 = solve_ivp(
                fun=lambda t, y: model.rhs_T1(
                    time=t,
                    time_max=model.years,
                    S1=model.S1,
                    S2=model.S2,
                    T1=y,
                    T2=model.T2,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=[y0_T1],
                **common_kwargs,
            )
            sol_T2 = solve_ivp(
                fun=lambda t, y: model.rhs_T2(
                    time=t,
                    time_max=model.years,
                    S1=model.S1,
                    S2=model.S2,
                    T1=model.T1,
                    T2=y,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=[y0_T2],
                **common_kwargs,
            )
        S1, S2 = sol_S1.y[0, :], sol_S2.y[0, :]
        if t_forcing is not None:
            T1, T2 = sol_T1.y[0, :], sol_T2.y[0, :]
    elif solution_cfg["integration_method"] == "odeint":
        common_kwargs = dict(
            t=teval,
            **solution_cfg.get("kwargs", {}),
        )
        sol_S1 = odeint(
            func=lambda y, t: model.rhs_S1(
                time=t,
                time_max=model.years,
                S1=y,
                S2=model.S2,
                T1=model.T1,
                T2=model.T2,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=model.S1,
            **common_kwargs,
        )
        sol_S2 = odeint(
            func=lambda y, t: model.rhs_S2(
                time=t,
                time_max=model.years,
                S1=model.S1,
                S2=y,
                T1=model.T1,
                T2=model.T2,
                fn_forcing=Fs_forcing,
                forcing_kwargs=s_forcing.forcing_kwargs,
            ),
            y0=model.S2,
            **common_kwargs,
        )
        if t_forcing is not None:
            sol_T1 = odeint(
                func=lambda y, t: model.rhs_T1(
                    time=t,
                    time_max=model.years,
                    S1=model.S1,
                    S2=model.S2,
                    T1=y,
                    T2=model.T2,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=model.T1,
                **common_kwargs,
            )
            sol_T2 = odeint(
                func=lambda y, t: model.rhs_T2(
                    time=t,
                    time_max=model.years,
                    S1=model.S1,
                    S2=model.S2,
                    T1=model.T1,
                    T2=y,
                    fn_forcing=Ft_forcing,
                    forcing_kwargs=t_forcing.forcing_kwargs,
                ),
                y0=model.T2,
                **common_kwargs,
            )
        S1, S2 = sol_S1, sol_S2
        if t_forcing is not None:
            T1, T2 = sol_T1, sol_T2

    Fs_plot = np.zeros(len(teval))
    Ft_plot = np.zeros(len(teval))
    qplot = np.zeros(len(teval))

    for i, t in enumerate(teval):
        Fs_plot[i] = Fs_forcing(t, model.years, **s_forcing.forcing_kwargs)
        if t_forcing is None:
            qplot[i] = model.q80(model.T1, model.T2, S1[i], S2[i])
        else:
            Ft_plot[i] = Ft_forcing(t, model.years, **t_forcing.forcing_kwargs)
            qplot[i] = model.q80(T1[i], T2[i], S1[i], S2[i])

    F_s = Fs_plot / Fs_to_m_per_year(model.S0, model.area)
    if t_forcing is not None:
        F_t = Ft_plot / Ft_to_m_per_year(model.T0, model.area)
    q = qplot / Sv

    series_dict = {
        "times": [teval, teval, teval, teval],
        "features": {
            "variables": {"S1": S1, "S2": S2},
            "forcings": {"Fs": F_s},
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
            },
            "forcings": {"Fs": r"\(F_s\)"},
        },
    }
    if t_forcing is not None:
        series_dict["features"]["variables"].update({"T1": T1, "T2": T2})
        series_dict["features"]["forcings"].update({"Ft": F_t})
        series_dict["latex"]["forcings"].update({"Ft": r"\(F_t\)"})
        series_dict["latex"]["variables"].update({"T1": r"\(T_1\)", "T2": r"\(T_2\)"})

    return series_dict


def get_time_series(
    model: BoxModel,
    s_forcing: Forcing,
    t_forcing: Optional[Forcing],
    nonlinear_density: bool,
    solution_cfg: DictConfig,
    init_time_series_cfg: DictConfig,
):
    series_dict = (
        get_time_series_nonlinear_(
            model=model,
            s_forcing=s_forcing,
            solution_cfg=solution_cfg,
            t_forcing=t_forcing,
            init_time_series_cfg=init_time_series_cfg,
        )
        if nonlinear_density
        else get_time_series_linear_(
            model=model,
            s_forcing=s_forcing,
            t_forcing=t_forcing,
            solution_cfg=solution_cfg,
            init_time_series_cfg=init_time_series_cfg,
        )
    )
    return series_dict


def plot_time_series(series_dict):
    variables = series_dict["features"]["variables"]
    forcings = series_dict["features"]["forcings"]
    latex_variables = series_dict["latex"]["variables"]
    latex_forcings = series_dict["latex"]["forcings"]
    units = series_dict["units"]
    time = series_dict["times"][0]

    xs_S = time / YEAR / 1000
    q = series_dict["q"]

    plt_kwargs = {
        "layout": "constrained",
    }
    fig_forcings, ax_forcings = plt.subplots(**plt_kwargs)
    fig_variables, ax_variables = plt.subplots(**plt_kwargs)
    fig_amoc, ax_amoc = plt.subplots(**plt_kwargs)

    x_label = r"\(\tau\) (kiloyears)"
    legend_kwargs = {
        "loc": "lower left",
        "bbox_to_anchor": (0, 1.02, 1, 0.2),
        "mode": "expand",
    }

    # Plot forcings
    ax_forcings.set_xlabel(x_label)
    for F_name, F in forcings.items():
        ax_forcings.plot(
            xs_S,
            F,
            label=f"{latex_forcings[F_name]} ({units[F_name]})",
        )
        ax_forcings.plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    # ax_forcings.set_box_aspect(1)
    # ax_forcings.legend(ncol=len(forcings), **legend_kwargs)

    # Plot the input variables
    ax_variables.set_xlabel(x_label)
    for var_name, value in variables.items():
        ax_variables.plot(
            xs_S,
            value,
            label=f"{latex_variables[var_name]} ({units[var_name]})",
        )
        ax_variables.plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    # ax_variables.set_box_aspect(1)
    # ax_variables.legend(ncol=len(variables), **legend_kwargs)

    # Plot q
    ax_amoc.plot(xs_S, q)
    ax_amoc.plot(xs_S, time * 0, "k--", dashes=(10, 5), lw=0.5)
    ax_amoc.set_xlabel(x_label)
    # ax_amoc.set_ylabel(rf"\(q\) ({units['q']})")
    ax_amoc.set_ylabel(f"AMOC ({units['q']})")
    # ax_amoc.set_box_aspect(1)

    # fig_forcings.legend(ncol=len(forcings), **legend_kwargs)
    # fig_variables.legend(ncol=len(variables), **legend_kwargs)
    # fig_forcings.tight_layout()
    # fig_variables.tight_layout()
    # fig_amoc.tight_layout()

    return fig_forcings, fig_variables, fig_amoc


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
