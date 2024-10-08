import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
import pyrenew.transformation as transformation
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.arrayutils import tile_until_n
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.distributions import CensoredNormal
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from pyrenew_hew.utils import get_vl_trajectory


class ww_site_level_dynamics_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        n_subpops,
        n_ww_lab_sites,
        unobs_time,
        n_initialization_points,
        max_shed_interval,
        i0_t_offset,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,
        eta_sd_rv,
        t_peak_rv,
        dur_shed_after_peak_rv,
        autoreg_rt_site_rv,
        sigma_rt_rv,
        i_first_obs_over_n_rv,
        sigma_i_first_obs_rv,
        sigma_initial_exp_growth_rate_rv,
        mean_initial_exp_growth_rate_rv,
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_hosp_mean_rv,
        p_hosp_w_sd_rv,
        autoreg_p_hosp_rv,
        hosp_wday_effect_rv,
        inf_to_hosp_rv,
        log10_g_rv,
        mode_sigma_ww_site_rv,
        sd_log_sigma_ww_site_rv,
        ww_site_mod_sd_rv,
        phi_rv,
        ww_ml_produced_per_day,
        pop_fraction,
        ww_uncensored,
        ww_censored,
        ww_sampled_lab_sites,
        ww_sampled_sites,
        ww_sampled_times,
        ww_log_lod,
        lab_site_to_site_map,
    ):  # numpydoc ignore=GL08
        self.state_pop = state_pop
        self.n_subpops = n_subpops
        self.n_ww_lab_sites = n_ww_lab_sites
        self.unobs_time = unobs_time
        self.n_initialization_points = n_initialization_points
        self.max_shed_interval = max_shed_interval
        self.i0_t_offset = i0_t_offset
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.t_peak_rv = t_peak_rv
        self.dur_shed_after_peak_rv = dur_shed_after_peak_rv
        self.autoreg_rt_site_rv = autoreg_rt_site_rv
        self.sigma_rt_rv = sigma_rt_rv
        self.i_first_obs_over_n_rv = i_first_obs_over_n_rv
        self.sigma_i_first_obs_rv = sigma_i_first_obs_rv
        self.sigma_initial_exp_growth_rate_rv = (
            sigma_initial_exp_growth_rate_rv
        )
        self.mean_initial_exp_growth_rate_rv = mean_initial_exp_growth_rate_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.p_hosp_w_sd_rv = p_hosp_w_sd_rv
        self.autoreg_p_hosp_rv = autoreg_p_hosp_rv
        self.hosp_wday_effect_rv = hosp_wday_effect_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.log10_g_rv = log10_g_rv
        self.mode_sigma_ww_site_rv = mode_sigma_ww_site_rv
        self.sd_log_sigma_ww_site_rv = sd_log_sigma_ww_site_rv
        self.ww_site_mod_sd_rv = ww_site_mod_sd_rv
        self.phi_rv = phi_rv
        self.ww_ml_produced_per_day = ww_ml_produced_per_day
        self.pop_fraction = pop_fraction
        self.ww_uncensored = ww_uncensored
        self.ww_censored = ww_censored
        self.ww_sampled_lab_sites = ww_sampled_lab_sites
        self.ww_sampled_sites = ww_sampled_sites
        self.ww_sampled_times = ww_sampled_times
        self.ww_log_lod = ww_log_lod
        self.lab_site_to_site_map = lab_site_to_site_map

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.ar_diff_rt = DifferencedProcess(
            fundamental_process=ARProcess(),
            differencing_order=1,
        )

        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        n_datapoints=None,
        data_observed_hospital_admissions=None,
        data_observed_log_conc=None,
    ):  # numpydoc ignore=GL08
        if n_datapoints is None and data_observed_hospital_admissions is None:
            raise ValueError(
                "Either n_datapoints or data_observed_hosp_admissions "
                "must be passed."
            )
        elif (
            n_datapoints is not None
            and data_observed_hospital_admissions is not None
        ):
            raise ValueError(
                "Cannot pass both n_datapoints and data_observed_hospital_admissions."
            )
        elif n_datapoints is None:
            n_datapoints = len(data_observed_hospital_admissions)
        else:
            n_datapoints = n_datapoints

        n_weeks_post_init = n_datapoints // 7 + 1

        eta_sd = self.eta_sd_rv()
        autoreg_rt = self.autoreg_rt_rv()
        log_r_mu_intercept = self.log_r_mu_intercept_rv()

        rt_init_rate_of_change_rv = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )

        rt_init_rate_of_change = rt_init_rate_of_change_rv()

        log_rtu_weekly = self.ar_diff_rt(
            noise_name="rtu_weekly_diff_first_diff_ar_process_noise",
            n=n_weeks_post_init,
            init_vals=jnp.array(log_r_mu_intercept),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
        )
        numpyro.deterministic("log_rtu_weekly", log_rtu_weekly)

        t_peak = self.t_peak_rv()
        # viral_peak = self.viral_peak_rv()
        dur_shed = self.dur_shed_after_peak_rv()

        s = get_vl_trajectory(t_peak, dur_shed, self.max_shed_interval)

        mean_initial_exp_growth_rate = self.mean_initial_exp_growth_rate_rv()
        sigma_initial_exp_growth_rate = self.sigma_initial_exp_growth_rate_rv()
        initial_exp_growth_rate_site_rv = TransformedVariable(
            "clipped_initial_exp_growth_rate_site",
            DistributionalVariable(
                "initial_exp_growth_rate_site_raw",
                dist.Normal(
                    mean_initial_exp_growth_rate,
                    sigma_initial_exp_growth_rate,
                ),
                reparam=LocScaleReparam(0),
            ),
            transforms=lambda x: jnp.clip(x, -0.01, 0.01),
        )

        i_first_obs_over_n = self.i_first_obs_over_n_rv()
        sigma_i_first_obs = self.sigma_i_first_obs_rv()
        i_first_obs_over_n_site_rv = TransformedVariable(
            "i_first_obs_over_n_site",
            DistributionalVariable(
                "i_first_obs_over_n_site_raw",
                dist.Normal(
                    transforms.logit(i_first_obs_over_n), sigma_i_first_obs
                ),
                reparam=LocScaleReparam(0),
            ),
            transforms=transforms.SigmoidTransform(),
        )

        with numpyro.plate("n_subpops", self.n_subpops):
            initial_exp_growth_rate_site = initial_exp_growth_rate_site_rv()
            i_first_obs_over_n_site = i_first_obs_over_n_site_rv()

        numpyro.deterministic(
            "initial_exp_growth_rate_site", initial_exp_growth_rate_site
        )

        log_i0_site = (
            jnp.log(i_first_obs_over_n_site)
            - self.unobs_time * initial_exp_growth_rate_site
        )
        numpyro.deterministic("log_i0_site", log_i0_site)

        autoreg_rt_site = self.autoreg_rt_site_rv()
        sigma_rt = self.sigma_rt_rv()
        rtu_site_ar_init_rv = DistributionalVariable(
            "rtu_site_ar_init",
            dist.Normal(
                0,
                sigma_rt / jnp.sqrt(1 - jnp.pow(autoreg_rt_site, 2)),
            ),
        )
        with numpyro.plate("n_subpops", self.n_subpops):
            rtu_site_ar_init = rtu_site_ar_init_rv()

        rtu_site_ar_proc = ARProcess()
        rtu_site_ar_weekly = rtu_site_ar_proc(
            noise_name="rtu_ar_proc",
            n=n_weeks_post_init,
            init_vals=rtu_site_ar_init[jnp.newaxis],
            autoreg=autoreg_rt_site[jnp.newaxis],
            noise_sd=sigma_rt,
        )

        numpyro.deterministic("rtu_site_ar_weekly", rtu_site_ar_weekly)

        rtu_site = jnp.repeat(
            jnp.exp(rtu_site_ar_weekly + log_rtu_weekly[:, jnp.newaxis]),
            repeats=7,
            axis=0,
        )[:n_datapoints, :]

        numpyro.deterministic("rtu_site", rtu_site)

        i0_site_rv = DeterministicVariable("i0_site", jnp.exp(log_i0_site))
        initial_exp_growth_rate_site_rv = DeterministicVariable(
            "initial_exp_growth_rate_site", initial_exp_growth_rate_site
        )

        infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            i0_site_rv,
            InitializeInfectionsExponentialGrowth(
                self.n_initialization_points,
                initial_exp_growth_rate_site_rv,
                t_pre_init=self.i0_t_offset,
            ),
        )

        generation_interval_pmf = self.generation_interval_pmf_rv()
        i0 = infection_initialization_process()
        numpyro.deterministic("i0", i0)

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc.sample(
            Rt=rtu_site,
            I0=i0,
            gen_int=generation_interval_pmf,
        )

        new_i_site = jnp.concat(
            [
                i0,
                inf_with_feedback_proc_sample.post_initialization_infections,
            ]
        )
        r_site_t = inf_with_feedback_proc_sample.rt
        numpyro.deterministic("r_site_t", r_site_t)

        state_inf_per_capita = jnp.sum(self.pop_fraction * new_i_site, axis=1)
        numpyro.deterministic("state_inf_per_capita", state_inf_per_capita)

        # number of net infected individuals shedding on each day (sum of individuals in diff stages of infection)
        def batch_colvolve_fn(m):
            return jnp.convolve(m, s, mode="valid")

        model_net_i = jax.vmap(batch_colvolve_fn, in_axes=1, out_axes=1)(
            new_i_site
        )[-n_datapoints:, :]
        numpyro.deterministic("model_net_i", model_net_i)

        log10_g = self.log10_g_rv()

        # expected observed viral genomes/mL at all observed and forecasted times
        model_log_v_ot = (
            jnp.log(10) * log10_g
            + jnp.log(model_net_i[:n_datapoints, :] + 1e-8)
            - jnp.log(self.ww_ml_produced_per_day)
        )
        numpyro.deterministic("model_log_v_ot", model_log_v_ot)

        # Hospital admission component
        p_hosp_mean = self.p_hosp_mean_rv()
        p_hosp_w_sd = self.p_hosp_w_sd_rv()
        autoreg_p_hosp = self.autoreg_p_hosp_rv()

        p_hosp_ar_init_rv = DistributionalVariable(
            "p_hosp_ar_init",
            dist.Normal(
                0,
                p_hosp_w_sd / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp, 2)),
            ),
        )

        p_hosp_ar_init = p_hosp_ar_init_rv()
        p_hosp_ar_proc = ARProcess()
        p_hosp_ar = p_hosp_ar_proc.sample(
            noise_name="p_hosp_noise",
            n=n_weeks_post_init,
            autoreg=autoreg_p_hosp,
            init_vals=p_hosp_ar_init,
            noise_sd=p_hosp_w_sd,
        )

        ihr = jnp.repeat(
            transformation.SigmoidTransform()(p_hosp_ar + p_hosp_mean),
            repeats=7,
        )[:n_datapoints]
        numpyro.deterministic("ihr", ihr)

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()
        inf_to_hosp = self.inf_to_hosp_rv()

        hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

        potential_latent_hospital_admissions = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=state_inf_per_capita,
                delay_incidence_to_observation_pmf=inf_to_hosp,
            )[-n_datapoints:]
        )

        latent_hospital_admissions = (
            potential_latent_hospital_admissions
            * ihr
            * hosp_wday_effect
            * self.state_pop
        )
        numpyro.deterministic(
            "latent_hospital_admissions", latent_hospital_admissions
        )

        mode_sigma_ww_site = self.mode_sigma_ww_site_rv()
        sd_log_sigma_ww_site = self.sd_log_sigma_ww_site_rv()
        ww_site_mod_sd = self.ww_site_mod_sd_rv()

        ww_site_mod_rv = DistributionalVariable(
            "ww_site_mod",
            dist.Normal(0, ww_site_mod_sd),
            reparam=LocScaleReparam(0),
        )  # lab-site specific variation

        sigma_ww_site_rv = TransformedVariable(
            "sigma_ww_site",
            DistributionalVariable(
                "log_sigma_ww_site",
                dist.Normal(jnp.log(mode_sigma_ww_site), sd_log_sigma_ww_site),
                reparam=LocScaleReparam(0),
            ),
            transforms=transforms.ExpTransform(),
        )

        with numpyro.plate("n_ww_lab_sites", self.n_ww_lab_sites):
            ww_site_mod = ww_site_mod_rv()
            sigma_ww_site = sigma_ww_site_rv()

        # Observations at the site level (genomes/person/day) are:
        # get a vector of genomes/person/day on the days WW was measured
        # These are the true expected genomes at the site level before observation error
        # (which is at the lab-site level)
        exp_obs_log_v_true = model_log_v_ot[
            self.ww_sampled_sites, self.ww_sampled_times
        ]

        # LHS log transformed obs genomes per person-day, RHS multiplies the expected observed
        # genomes by the site-specific multiplier at that sampling time
        exp_obs_log_v = (
            exp_obs_log_v_true + ww_site_mod[self.ww_sampled_lab_sites]
        )

        hospital_admission_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions", concentration_rv=self.phi_rv
        )

        observed_hospital_admissions = hospital_admission_obs_rv(
            mu=latent_hospital_admissions,
            obs=data_observed_hospital_admissions,
        )

        numpyro.sample(
            "log_conc_obs",
            CensoredNormal(
                loc=exp_obs_log_v,
                scale=sigma_ww_site[self.ww_sampled_lab_sites],
                lower_limit=self.ww_log_lod,
            ),
            obs=data_observed_log_conc,
        )

        ww_pred = numpyro.sample(
            "site_ww_pred",
            dist.Normal(
                loc=model_log_v_ot[:, self.lab_site_to_site_map] + ww_site_mod,
                scale=sigma_ww_site,
            ),
        )

        state_model_net_i = jnp.convolve(
            state_inf_per_capita, s, mode="valid"
        )[-n_datapoints:]
        numpyro.deterministic("state_model_net_i", state_model_net_i)

        state_log_c = (
            jnp.log(10) * log10_g
            + jnp.log(state_model_net_i[:n_datapoints] + 1e-8)
            - jnp.log(self.ww_ml_produced_per_day)
        )
        numpyro.deterministic("state_log_c", state_log_c)

        exp_state_ww_conc = jnp.exp(state_log_c)
        numpyro.deterministic("exp_state_ww_conc", exp_state_ww_conc)

        state_rt = (
            state_inf_per_capita[-n_datapoints:]
            / jnp.convolve(
                state_inf_per_capita,
                jnp.hstack(
                    (jnp.array([0]), jnp.array(generation_interval_pmf))
                ),
                mode="valid",
            )[-n_datapoints:]
        )
        numpyro.deterministic("state_rt", state_rt)

        return (
            latent_hospital_admissions,
            observed_hospital_admissions,
            ww_pred,
        )
