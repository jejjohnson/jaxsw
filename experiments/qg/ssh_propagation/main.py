import autoroot
import hydra
import pde
import jax
import jax.numpy as jnp
import xarray as xr
import pandas as pd
import diffrax as dfx
from loguru import logger
from jaxsw._src.domain.time import TimeDomain
import time

# from jax.config import config

# config.update("jax_enable_x64", True)


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg):
    logger.info(f"Starting QG experiment...")

    logger.info(f"Loading SSH Dataset...")
    ssh_init = xr.open_dataset(cfg.data.filename).ssh[0]

    t0_semantic = ssh_init.time.values
    logger.info(f"Starting at: {t0_semantic}")

    logger.info(f"Choosing Time Steps... ")
    t0 = hydra.utils.instantiate(cfg.timestepper.tmin)
    t1 = hydra.utils.instantiate(cfg.timestepper.tmax)
    dt = hydra.utils.instantiate(cfg.timestepper.dt)
    dtsave = hydra.utils.instantiate(cfg.timestepper.tsave)

    logger.info(f"T0 {int(t0/60/60/24)} [days]")
    logger.info(f"T1 {int(t1/60/60/24)} [days]")
    logger.info(f"DT {int(dt/60)} [minutes]")
    logger.info(f"TSave {int(dtsave/60/60)} [hours]")

    t_domain = TimeDomain(tmin=t0, tmax=t1, dt=dt)
    ts = jnp.arange(t0, t1 + dtsave, dtsave)
    saveat = dfx.SaveAt(ts=ts)

    # choose solver
    logger.info(f"Initializing solver...")
    solver = hydra.utils.instantiate(cfg.solver)

    logger.info(f"Initializing stepsize controller...")
    stepsize_controller = hydra.utils.instantiate(cfg.controller)

    # dynamical system
    logger.info(f"Initializing PDE...")
    dyn_model = pde.QG(
        t_domain=t_domain,
        saveat=saveat,
        solver=solver,
        stepsize_controller=stepsize_controller,
    )

    logger.info(f"Initializing state...")
    state_init, params = pde.State.init_state(ssh_init, c1=cfg.model.c1)

    logger.info(f"Test run...")
    state_out = pde.QG.equation_of_motion(0, state_init, params)
    logger.info(f"Success..!")

    logger.info(f"Starting Integration...")
    tstart = time.time()
    state_sol = dyn_model.integrate(
        state_init, dt, params, max_steps=cfg.timestepper.max_steps
    )
    tfin = time.time() - tstart
    logger.info(f"Finished in {tfin:.2f} [s]...!")

    logger.info(f"Calculating SSH from Saved States...")
    ssh_t = jax.vmap(pde.QG.ssh_from_state, in_axes=(0, None))(state_sol, params)

    logger.info(f"Creating xr.Dataset...")
    ts_semantic = t0_semantic + pd.to_timedelta(ts, unit="seconds")

    out_ds = xr.Dataset(
        {
            "ssh": (("time", "lon", "lat"), ssh_t),
            "pvort": (("time", "lon", "lat"), state_sol.q),
        },
        coords={
            "time": (("time"), ts_semantic),
            "lat": ssh_init.lat,
            "lon": ssh_init.lon,
        },
    )
    out_ds = out_ds.transpose("time", "lat", "lon")
    logger.info(f"Saving...")
    out_ds.to_netcdf(cfg.data.save_name)

    logger.info(f"Done experiment...!")


if __name__ == "__main__":
    main()
