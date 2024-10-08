import jax.numpy as jnp


def convert_to_logmean_log_sd(mean, sd):
    logmean = jnp.log(
        jnp.power(mean, 2) / jnp.sqrt(jnp.power(sd, 2) + jnp.power(mean, 2))
    )
    logsd = jnp.sqrt(jnp.log(1 + (jnp.power(sd, 2) / jnp.power(mean, 2))))
    return logmean, logsd


def normed_shedding_cdf(
    time: float, t_p: float, t_d: float, log_base: float
) -> float:
    """
    fraction of total fecal RNA shedding that has occurred
    by a given time post infection.
    """
    norm_const = (t_p + t_d) * ((log_base - 1) / jnp.log(log_base) - 1)
    ad_pre = (
        lambda x: t_p
        / jnp.log(log_base)
        * jnp.exp(jnp.log(log_base) * x / t_p)
        - x
    )
    ad_post = (
        lambda x: -t_d
        / jnp.log(log_base)
        * jnp.exp(jnp.log(log_base) * (1 - ((x - t_p) / t_d)))
        - x
    )
    return (
        jnp.where(
            time < t_p + t_d,
            jnp.where(
                time < t_p,
                ad_pre(time) - ad_pre(0),
                ad_pre(t_p) - ad_pre(0) + ad_post(time) - ad_post(t_p),
            ),
            norm_const,
        )
        / norm_const
    )


def get_vl_trajectory(tpeak, duration_shedding_after_peak, max_days):
    daily_shedding_pmf = normed_shedding_cdf(
        jnp.arange(1, max_days), tpeak, duration_shedding_after_peak, 10
    ) - normed_shedding_cdf(
        jnp.arange(0, max_days - 1), tpeak, duration_shedding_after_peak, 10
    )
    return daily_shedding_pmf
