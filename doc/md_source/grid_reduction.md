# 👾 Grid Reduction

VeraGrid has the ability to perform planning-style grid reductions.

A) From the schematic:

1. First, Select the buses that you want to reduce in the schematic.

2. The launch the grid reduction window by selecting the menu option `Model > Grid Reduction`

B) From the database:

1. Select the buses according to some rule in the database view. 
2. Right click and call the **grid reduction** action in the context menu.

![](figures/grid_reduction_1.png)


A small window  will pop up indicating the list of buses that you are going to remove.
By accepting, the grid will be reduced according to the theory developed by [1].
You can expect that the reduced grid flows behave roughly like the original grid.

Changing the injections, or further topological changes will alter the equivalent behavior.

This action cannot be undone.

## API

Grid reduction examples:

```python
import numpy as np
import VeraGridEngine as vg

grid = vg.open_file('case118.m')

pf_res = vg.power_flow(grid=grid)

# define the buses to remove
reduction_bus_indices = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 113, 114, 115, 117
]) - 1  # minus 1 for zero based indexing

# Ward reduction
grid_ward = vg.ward_standard_reduction(
    grid=grid.copy(),
    reduction_bus_indices=reduction_bus_indices,
    V0=pf_res.voltage,
    logger=vg.Logger()
)

# PTDF reduction
nc = vg.compile_numerical_circuit_at(circuit=grid, t_idx=None)
lin = vg.LinearAnalysis(nc=nc)

if grid.has_time_series:
    lin_ts = vg.LinearAnalysisTs(grid=grid)
else:
    lin_ts = None

grid_ptdf, logger_ptdf = vg.ptdf_reduction(
    grid=grid.copy(),
    reduction_bus_indices=reduction_bus_indices,
    PTDF=lin.PTDF,
    lin_ts=lin_ts
)

# Di-Shi reduction
grid_di_shi, logger_ds = vg.di_shi_reduction(
    grid=grid.copy(),
    reduction_bus_indices=reduction_bus_indices,
    V0=pf_res.voltage
)

```

Observe how we feed a copy of the original grid to the 
reduction functions. This is because those functions alter
the input grid.

## Theory


### Ward reduction

Performs the standard ward reduction.

Define the bus sets $E$ (external buses to remove), 
$I$ (internal buses that are not boundary) and 
$B$ (boundary buses)

- Run a power flow of the base grid and slice $V_b = V[B]$ and $V_e = V[E]$
- Slice $Y_{BE} = Y[B, E]$, $Y_{EB} = Y[E, B]$ and $Y_{EE} = Y[E, E]$
- Compute the equivalent boundary admittances as:
$$
Y_{eq} = Y_{BE} \times Y_{EE}^{-1} \times Y_{EB}
$$
- Compute the equivalent boundary injection currents as: 
$$
I_{eq} = - Y_{BE} \times Y_{EE}^{-1} \times (Y_{EB} \times V_b + Y_{EE} \times V_e)
$$

- Compute the boundary power injections as:
$$
S_{eq} = V_b \cdot (I_{eq})^*
$$
- Create a new load with the value of $S_{eq}[b]$ for every bus $b$ of $B$.
- For every entry in the lower triangle of $Y_{eq}$, create a shunt or series reactance 
at the boundary or between the boundary buses.
- Finally, remove all buses in $E$ from the grid.

### PTDF reduction

Performs a simple PTDF-based projection of the 
injections in the external grid into the boundary buses.
This does not respect the pre-existing devices integrity.

Along with the bus sets $E$ (external buses to remove), 
$I$ (internal buses that are not boundary) and 
$B$ (boundary buses), we add a set of branches that 
join an external bus to a boundary bus and call if $BE$.

- For every injection device connected to a bus $e$ of $E$:

  - For every bus $b$ of $B$:
  
    - get the branch from $BE$ associated to the bus $b$
    - get the PTDF value of the branch and boundary bus $PTDF[BE_b, b]$
    - Create a new injection device connected to $b$ with the projected power $P_{proj} = P \cdot PTDF[BE_b, b]$

- Finally, remove all buses in $E$ from the grid.

### Di-Shi grid equivalent

The PhD dissertation of Di-Shi presented in [1] [2], expands on the traditional ward equivalent reduction method.
The proposed method allows the generators to be *just moved* to the boundary buses. Later, 
the injections are calibrated to compensate for that. It is a very friendly method for 
planning engineers that want to reduce the grid, and still need to keep the generators as previously 
defined for dispatching.

**Step 0 – Define bus sets**

- I: set of internal buses.
- E: set of external buses: those that we want to remove.
- B: set of boundary buses between E and I.

**Step 1 – First Ward reduction**

This first reduction is to obtain the equivalent admittance matrix $Y_eq^{(1}$ that serves
to create the inter-boundary branches that represent the grid that we are going to remove.
For this the buses to keep are the internal (I) + boundary (B).

**Step 2 – Second Ward reduction: Extending to the external generation buses**

The second reduction is to generate another equivalent admittance matrix $Y_eq^{(2}$
that we use as adjacency matrix to search the closest bus to move each generator that is external.
For this the buses to keep are the internal (I) + boundary (B) + the generation buses of E.

**Step 3 – Relocate generators**

Using the matrix $Y_eq^{(2}$, we calculate the shortest paths from every 
external generation bus, to all the other buses in I + B. The end of each 
path will be the relocation bus of every external generator.

**Step 4 – Relocate loads with inverse power flow**

Let's not forget about the loads! in order to move the external loads such that
the reduced flows resemble the original flows (even after brutally moving the generators!),
we need to perform an *inverse power flow*.

First, we need to run a linear power flow in the original system. 
That will get us the original voltage angles.

Second, we need to form the admittance matrix of the reduced grid 
(including the inter-boundary branches), and multiply this admittance
matrix by the original voltage angles for the reduced set of buses.
This gets us the "final" power injections in the reduced system.

From those, we need to subtract the reduced grid injections. 
This will provide us with a vector of new loads that we need to add at 
the corresponding reduced grid buses in order to have a final equivalent.


[1]: [Power System Network Reduction for Engineering and Economic 
Analysis by Di Shi, 
2012 Arizona State University](https://core.ac.uk/download/pdf/79564835.pdf).

[2]: [Optimal Generation Investment Planning: Pt 1: Network Equivalents](https://ieeexplore.ieee.org/document/6336375)
