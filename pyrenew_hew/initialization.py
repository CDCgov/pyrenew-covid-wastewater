import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from pyrenew_hew.utils import convert_to_logmean_log_sd


def get_initialization(stan_data, stdev, rng_key):
    i_first_obs_est = (
        np.mean(stan_data["hosp"][:7]) / stan_data["p_hosp_prior_mean"]
    )
    logit_i_frac_est = jax.scipy.special.logit(
        i_first_obs_est / stan_data["state_pop"]
    )

    init_vals = {
        "eta_sd": jnp.abs(dist.Normal(0, stdev).sample(rng_key)),
        "autoreg_rt": jnp.abs(
            dist.Normal(
                stan_data["autoreg_rt_a"]
                / (stan_data["autoreg_rt_a"] + stan_data["autoreg_rt_b"]),
                0.05,
            ).sample(rng_key)
        ),
        "log_r_mu_intercept": dist.Normal(
            convert_to_logmean_log_sd(1, stdev)[0],
            convert_to_logmean_log_sd(1, stdev)[1],
        ).sample(rng_key),
        "sigma_rt": jnp.abs(dist.Normal(0, stdev).sample(rng_key)),
        "autoreg_rt_site": jnp.abs(dist.Normal(0.5, 0.05).sample(rng_key)),
        "sigma_i_first_obs": jnp.abs(dist.Normal(0, stdev).sample(rng_key)),
        "sigma_initial_exp_growth_rate": jnp.abs(
            dist.Normal(0, stdev).sample(rng_key)
        ),
        "i_first_obs_over_n": jax.nn.sigmoid(
            dist.Normal(logit_i_frac_est, 0.05).sample(rng_key)
        ),
        "mean_initial_exp_growth_rate": dist.Normal(0, stdev).sample(rng_key),
        "inv_sqrt_phi": 1 / jnp.sqrt(200)
        + dist.Normal(1 / 10000, 1 / 10000).sample(rng_key),
        "mode_sigma_ww_site": jnp.abs(
            dist.Normal(
                stan_data["mode_sigma_ww_site_prior_mode"],
                stdev * stan_data["mode_sigma_ww_site_prior_sd"],
            ).sample(rng_key)
        ),
        "sd_log_sigma_ww_site": jnp.abs(
            dist.Normal(
                stan_data["sd_log_sigma_ww_site_prior_mode"],
                stdev * stan_data["sd_log_sigma_ww_site_prior_sd"],
            ).sample(rng_key)
        ),
        "p_hosp_mean": dist.Normal(
            jax.scipy.special.logit(stan_data["p_hosp_prior_mean"]), stdev
        ).sample(rng_key),
        "p_hosp_w_sd": jnp.abs(dist.Normal(0.01, 0.001).sample(rng_key)),
        "autoreg_p_hosp": jnp.abs(dist.Normal(1 / 100, 0.001).sample(rng_key)),
        "t_peak": dist.Normal(
            stan_data["viral_shedding_pars"][0],
            stdev * stan_data["viral_shedding_pars"][1],
        ).sample(rng_key),
        "viral_peak": dist.Normal(
            stan_data["viral_shedding_pars"][2],
            stdev * stan_data["viral_shedding_pars"][3],
        ).sample(rng_key),
        "dur_shed": dist.Normal(
            stan_data["viral_shedding_pars"][4],
            stdev * stan_data["viral_shedding_pars"][5],
        ).sample(rng_key),
        "log10_g": dist.Normal(stan_data["log10_g_prior_mean"], 0.5).sample(
            rng_key
        ),
        "ww_site_mod_sd": jnp.abs(dist.Normal(0, stdev).sample(rng_key)),
        "hosp_wday_effect_raw": jax.nn.softmax(
            jnp.abs(dist.Normal(1 / 7, stdev).expand([7]).sample(rng_key))
        ),
        "inf_feedback_raw": jnp.abs(dist.Normal(500, 20).sample(rng_key)),
    }

    return init_vals
