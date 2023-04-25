

**Simple Run**

*This is the fastest run* because we use an Euler integration scheme.

```bash
python main.py
```

**Configuration From Redouane**

*Redouane* used a similar method for his simulations with a specific time stepping configuration.

```bash
python main.py ++timestepper.tmax.arg=20 ++timestepper.dt.arg=600 ++timestepper.dt.unit="seconds"
```


**Configuration From Florian**

*This is a slower run* because we use a Runge-Kutta-like integration scheme.

```bash
python main.py solver=Dopri5 
```



**Advanced Configuration**

*Here we add a lot of bells and whistles with some special stepsize controllers*

```bash
python main.py data=jeanzay ++timestepper.tmax.freq=10 ++timestepper.dt.freq=600 ++timestepper.dt.unit="seconds" solver=tsit5 controller=adaptive ++controller.dtmax.freq=6
```