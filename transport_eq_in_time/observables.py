import importlib, enum
from dataclasses import dataclass, field
import numpy as np, matplotlib.pyplot as plt
from typing import Any, Tuple
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import root
import decay_process, axion_motion, transport_equation
# decay_process, axion_motion, transport_equation = map(importlib.reload, (decay_process, axion_motion, transport_equation))

def calc_entropy_density(T, g_star):
    return 2*np.pi**2 / 45 * g_star * T**3

zeta3 = 1.20206
g_photon = 2
C_sph = 8 / 23
eta_B_observed = 6e-10 # from paper
g_star_0 = 43/11 # from paper
asym_const = - g_star_0 * C_sph * np.pi**2 / (6 * decay_process.g_star * zeta3 * g_photon)
T_CMB = 2.348654180597668e-13 # GeV
s_today = calc_entropy_density(T_CMB, g_star_0)
h = 0.673
rho_c = 3.667106289005098e-11 # [eV^4]
Omega_DM_h_sq = 0.11933

def abundance_to_relic_density(Y, m):
    n = s_today * Y
    rho = m * n
    Omega_h_sq = rho * 1e9**4 / rho_c * h**2
    return Omega_h_sq

def red_chem_pot_to_asymmetry(red_chem_pot_B_minus_L):
    return asym_const * red_chem_pot_B_minus_L

Status = enum.Enum("Status", """
OK
ASYM_CONVERGENCE_FAILURE
RELIC_DENSITY_CONVERGENCE_FAILURE
AXION_OSCILLATES_BEFORE_INFLATION
INFLATON_DECAYS_DURING_INFLATION
ISOCURVATURE_BOUNDS
""")

@dataclass
class AsymmetrySolverConfig:
    calc_init_time: bool = False
    asym_max_steps: int|None = None
    axion_decay_time: float = 10.0
    return_evolution: bool = False
    nsamples: int = 100
    debug: bool = False
    isocurvature_check: bool = False
    rtol_asym: float = 1e-3
    step_tmax_axion_time: float = 2*2*np.pi
    T_goal: float = 1e12 # [GeV]
    start_step_axion: float = 1.0 + 10.0*2*np.pi

@dataclass
class RelicDensitySolverConfig:
    nosc_per_step: int = 5
    nsamples_per_osc: int = 20
    rtol_relic: float = 1e-3
    relic_max_steps: int|None = None
    debug: bool = False

@dataclass
class Model:
    H_inf: float
    Gamma_inf: float
    axion_parameter: ArrayLike
    axion_model: axion_motion.AxionField
    source_vector_axion: NDArray[float]
    f_a: float
    energy_scale: float
    conv_factor: float
    scale: float

    @classmethod
    def make(cls, H_inf, Gamma_inf, axion_model, axion_parameter, source_vector, f_a):
        energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
        conv_factor = Gamma_inf / energy_scale
        scale = decay_process.find_scale(Gamma_inf)
        return cls(H_inf, Gamma_inf, axion_model, axion_parameter, source_vector, f_a,
                energy_scale, conv_factor, scale)

    def axion_to_inf_time(self, t_axion: float) -> float:
        return t_axion * self.conv_factor

    def inf_to_axion_time(self, t_inf: float) -> float:
        return t_inf / self.conv_factor

    def check(self, asym_config: AsymmetrySolverConfig):
        if self.energy_scale > self.H_inf:
            return Status.AXION_OSCILLATES_BEFORE_INFLATION.value #(invalidates the assumtion that the axion doesn't iterfere with inflation)
        if self.Gamma_inf > self.H_inf:
            return Status.INFLATON_DECAYS_DURING_INFLATION.value
        if asym_config.isocurvature_check and self.H_inf / (2*np.pi) / self.f_a < 1e-5: # eq. 1 in 1412.2043
            return Status.ISOCURVATURE_BOUNDS.value
        return Status.OK

    def get_initial_integration_time(state: "State", asym_config: AsymmetrySolverConfig):
        if asym_config.calc_init_time:
            T_max = (
                0.8 * decay_process.g_star**(-1/4) * state.rho_inf**(1/8)
                * (self.Gamma_inf * decay_process.M_pl * np.sqrt(8*np.pi))**(1/4)
            ) # [GeV]
            if asym_config.T_goal < T_max:
                # asym_config.T_goal is never reached
                return 1.0 # advance to RH
            else:
                T_RH = 0.55 * g_star**(-1/4)*(M_Pl * np.sqrt(8 * np.pi) * self.Gamma_inf)**(1/2)
                # figure out when T(t) = T_eq_general
                if asym_config.T_goal > T_RH:
                    # asym_config.T_goal is in radiation domination
                    H_goal = np.sqrt(np.pi**2/30*g_star*asym_config.T_goal**4 / (3*M_pl**2))
                    return (1/(H_goal / Gamma_inf) - 1/(H_inf / Gamma_inf)) / 2 + decay_process.t0
                else:
                    # asym_config.T_goal is in inflaton oscillation domaintion
                    return 1.0 # advance to T_RH
        else:
            return self.axion_to_inf_time(asym_config.start_step_axion)


@dataclass
class State:
    t_inf: float
    t_axion: float
    rho_rad: float
    rho_inf: float
    red_chem_pots: NDArray[float]
    axion: NDArray[float]

    @classmethod
    def initial(cls, model: Model, axion_init: NDArray[float]) -> "State":
        return cls(t_inf=decay_process.t0, t_axion=0.0,
                rho_rad=0.0, rho_inf=3 * decay_process.M_pl**2 * model.H_inf**2,
                red_chem_pots=np.zeros(transport_equation.N), axion=axion_init)

    def advance(self, model: Model, Delta_t_axion: float, asym_config: AsymmetrySolverConfig) -> "Solution":
        Delta_t_inf = model.axion_to_inf_time(Delta_t_axion)

        ############################### background cosmology i.e. reheating ##################################
        sol_rh, T_and_H_fn, T_and_H_and_T_dot_fn = decay_process.solve(Delta_t_inf, self.rho_rad, self.rho_inf, model.scale, model.Gamma_inf)

        ############################### evolution of the axion field ##################################
        sol_axion = axion_model.solve(self.axion, model.axion_parameter, Delta_t_axion, T_and_H_fn, model.Gamma_inf)
        axion_source = axion_model.get_source(sol_axion, model.conv_factor, *model.axion_parameter)

        ######################### transport eq. for standard model charges ##############################
        sol_transp_eq = transport_equation.solve(Delta_t_inf, self.red_chem_pots, T_and_H_and_T_dot_fn,
                    axion_source, model.source_vector_axion, model.Gamma_inf, model.conv_factor)

        return Solution(model=model, initial_state=self,
                sol_rh=sol_rh, T_and_H_fn=T_and_H_fn, T_and_H_and_T_dot_fn=T_and_H_and_T_dot_fn,
                sol_axion=sol_axion, axion_source=axion_source, sol_transp_eq=sol_transp_eq,
                tmax_axion_time=Delta_t_axion, tmax_inf_timetmax_inf_time=model.axion_to_inf_time(Delta_t_axion))

    def get_asymmetry(self):
        return red_chem_pot_to_asymmetry(transport_equation.calc_B_minus_L(self.red_chem_pots))

@dataclass
class Solution:
    model: Model
    initial_state: State
    sol_rh: Any
    T_and_H_fn: Any
    T_and_H_and_T_dot_fn: Any
    sol_axion: Any
    axion_source: Any
    sol_transp_eq: Any
    tmax_axion_time: float
    tmax_inf_time: float

    def get_final_state(self) -> State:
        return State(t_inf=tmax_inf_time, t_axion=tmax_axion_time,
                rho_rad=decay_process.find_end_rad_energy(self.sol_rh, self.model.scale),
                rho_inf=decay_process.find_end_field_energy(self.sol_rh, self.initial_state.rho_inf),
                red_chem_pots=self.sol_transp_eq(np.log(self.tmax_inf_time)),
                axion=self.sol_axion(tmax_axion_time),
        )

    def is_asymmetry_converged(self, asym_config: AsymmetrySolverConfig):
        log_ts_inf = np.linspace(np.log(decay_process.t0), np.log(decay_process.t0 + self.tmax_inf_time), asym_config.nsamples)
        red_chem_pots = self.sol_transp_eq(log_ts_inf) # returned function converts internal unit
        B_minus_L_red_chem = transport_equation.calc_B_minus_L(red_chem_pots)
        # convergence by change within the last integration interval
        a, b = np.max(B_minus_L_red_chem), np.min(B_minus_L_red_chem)
        delta = np.abs((a - b) / np.mean(B_minus_L_red_chem))

        if asym_config.debug:
            print("B-L start .. end:", B_minus_L_red_chem[0], B_minus_L_red_chem[-1])
            print("B-L range:", b, a)
            print("delta =", delta, "rtol_asym =", rtol_asym)

        return delta < asym_config.rtol_asym


def compute_asymmetry(model: Model, asym_config: AsymmetrySolverConfig) -> Tuple[float, Status]|list[Solution]:
    step = 1
    sols = []
    status = model.ckeck(asym_config)
    if status != Status.OK:
        return np.nan, status
    Delta_t_axion = model.get_initial_integration_time()

    while True:
        if asym_config.debug:
            print("step =", step)

        sol = state.advance(Delta_t_axion, asym_config)
        if asym_config.return_evolution:
            sols.append(sol)
        state = sol.get_final_state()

        if sol.is_asymmetry_converged():
            break

        if asym_config.asym_max_steps is not None and step > asym_config.asym_max_steps:
            status = Status.ASYM_CONVERGENCE_FAILURE
            break

        Delta_t_axion = asym_config.step_tmax_axion_time
        step += 1

    if return_evolution:
        return status, sols
    else:
        return status, state, state.get_asymmetry()


def compute_dilution_factor_from_axion_decay(model: Model, state: State, asym_config: AsymmetrySolverConfig) -> float:
    rho_axion = model.axion_model.get_energy(state.axion, model.f_a, model.axion_parameter)
    if not (np.isfinite(state.rho_rad) and np.isfinite(rho_axion)):
        return np.nan
    # dilution factor from axion decay
    # we don't do converence check for this part right now
    Gamma_axion = model.axion_model.get_decay_constant(model.f_a, *model.axion_parameter) # [GeV]
    axion_scale = decay_process.find_scale(Gamma_axion) # GeV
    sol_axion_decay, T_and_H_fn_axion, _ = \
            decay_process.solve(asym_config.axion_decay_time, state.rho_rad, rho_axion, axion_scale, Gamma_axion)
    t = np.exp(sol_axion_decay.t[-1])
    f = decay_process.find_dilution_factor(sol_axion_decay, T_and_H_fn_axion, t)
    return f

def compute_relic_density_from_state(mode: Model, state: State, relic_config: RelicDensitySolverConfig):
    assert state.rho_inf / state.rho_rad < 1e-5
    H_start = np.sqrt(state.rho_rad / (3 * M_pl**2))
    return compute_relic_density(mode.axion_model, model.axion_parameter, state.axion, model.f_a, H_start)

def compute_relic_density(axion_model: axion_motion.AxionField, axion_parameter, axion_initial,
                          f_a, H_start, relic_config: RelicDensitySolverConfig):
    energy_scale = axion_model.find_dynamical_scale(*axion_parameter)
    H_start /= energy_scale
    axion_state = axion_initial

    def make_T_and_H_fn(H0):
        def T_and_H_fn(t):
            H = 1 / (2*t + 1 / H0)
            T = (g_start * np.pi**2 / 90 / M_pl**2 / H**2)**(1/4)
            return T, H

    current_t = 0
    H_osc = axion_model.find_H_osc(*axion_parameter)
    fake_Gamma_inf = 1.0

    ################ evolve to oscillation start, if nessesary ####################
    if H_start > H_osc:
        current_t = t_osc = 1 / 2 * (1/H_osc - 1/H_start)
        if debug:
            print("advancing to oscillation: from {H_start =} to {H_osc =} integrating {t_osc =}")
        T_and_H_fn = make_T_and_H_fn(H_start)
        sol_axion = axion_model.solve(axion_state, axion_parameter, t_osc, T_and_H_fn, fake_Gamma_inf)
        axion_state = sol_axion.y[:, -1]

    ####################### convergence of axion abundance ####################
    last_Y_estimate = np.nan
    step = 1

    while True:
        ######## advance background cosmology and axion field ###########
        T, H = T_and_H_fn(current_t)
        T_and_H_fn = make_T_and_H_fn(H)
        freq = axion_model.find_mass(current_T, *axion_parameter)
        step_time_axion = nosc_per_step * 2*np.pi / freq
        sol_axion = axion_model.solve(axion_state, axion_parameter, step_time_axion, T_and_H_fn, fake_Gamma_inf)
        axion_state = sol_axion.y[:, -1]
        current_t += step_time_axion

        ############ convergence check ############
        ts = np.linspace(0, step_time_axion, nsamples_per_osc * nosc_per_step)
        T, H = T_and_H_fn(ts)
        s = calc_entropy_density(T, decay_process.g_star) # [GeV^3]
        ys = sol_axion.sol(ts)
        fake_f_a = 1.0
        rho_over_f_sq = np.array([axion_model.get_energy(y, fake_f_a, *axion_parameter) for y in ys.T])
        m = axion_model.find_mass(T, *axion_parameter)
        n_over_f_sq = rho_over_f_sq / m # [GeV]
        Y = n_over_f_sq / s # [GeV^-2] bc its n/s/f^2
        is_min = (Y[2:] > Y[1:-1]) & (Y[:-2] > Y[1:-1])
        is_max = (Y[2:] < Y[1:-1]) & (Y[:-2] < Y[1:-1])
        Y_estimate = (np.sum(Y[1:-1][is_min]) + np.sum(Y[1:-1][is_max])) / (np.sum(is_min) + np.sum(is_max))
        delta = np.abs(Y_estimate - last_Y_estimate) / Y_estimate
        if debug:
            print(f"{step =}: {delta =} vs {rtol_relic =}")
        if delta < rtol_relic:
            break
        if relic_max_steps is not None and step > relic_max_steps:
            return Status.RELIC_DENSITY_CONVERGENCE_FAILURE, np.nan
        last_Y_estimate = Y_estimate
        step += 1

    ######### finish relic density ###########
    m_today = axion_model.find_mass(T_CMB / energy_scale, *axion_parameter)
    Omega_h_sq = abundance_to_relic_density(Y_estimate, m_today) * f_a**2
    return Status.OK, Omega_h_sq



