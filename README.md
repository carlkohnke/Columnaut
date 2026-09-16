# Columnaut

A multicomponent distillation-column simulator for predicting steady-state
product compositions, stage temperatures, internal flowrates, and
condenser/reboiler duties from a specified column design, feed, and operating
conditions.

The solver uses a sequential bubble-point method, Peng-Robinson thermodynamics,
and banded material-balance solves.

## Install

```text
git clone https://github.com/carlkohnke/Columnaut.git
cd Columnaut
python -m venv .venv
python -m pip install -e ".[test]"
```

Activate the virtual environment first if you do not want to use your system
Python. Python 3.11 or newer is required.

## Dependencies

Runtime dependencies are NumPy, SciPy, PyYAML, and Matplotlib.

## Run an example simulation

Six-component, 30-tray hydrocarbon column:

```text
python -m distillation.cli simulate examples/hydrocarbon_column.yaml --output results/hydrocarbon
```

Smaller three-component column:

```text
python -m distillation.cli simulate examples/three_component_column.yaml --output results/three_component
```

Select one or more reflux ratios by repeating `--reflux-ratio`:

```text
python -m distillation.cli simulate examples/hydrocarbon_column.yaml --reflux-ratio 2.0 --reflux-ratio 2.5 --output results/selected
```

Add `--no-plots` when only JSON, NPZ, and CSV results are needed.

## Python API

```python
from distillation import load_config, simulate

config = load_config("examples/hydrocarbon_column.yaml")
batch = simulate(config)

for result in batch.results:
    print(result.reflux_ratio, result.iterations)
    print(result.temperature_k)
    print(result.diagnostics["max_bubble_point_residual"])

batch.save("results/from_api")
```

For direct control, construct `ColumnSolver(config, thermodynamics=package)`.
The column solver depends on the `ThermodynamicPackage` interface rather than
Peng-Robinson implementation details.

## Engineering stage convention

All configuration and output files use one-based engineering stage numbers:

- stage 1 is the total condenser;
- stages 2 through `equilibrium_trays + 1` are column trays;
- the final stage is the partial reboiler.

Python array indexing is converted internally and is never exposed in the input
schema. Feed-stage validation reports errors using engineering stage numbers.

### Adding a chemical

Copy an existing entry in `component_database.yaml`, choose a unique `name`,
and supply all fields shown in that file. The new name can then be selected in
any column configuration.

## What this program calculates

The main inputs are:

- selected chemicals and their thermodynamic properties;
- number of equilibrium trays and feed-stage locations;
- stage pressures;
- feed flowrate, composition, temperature, pressure, and vapor fraction;
- reflux ratio;
- either the distillate flow, bottoms flow, or the `perfect_split` flow
  estimate used to close the overall balance.

The main outputs are the tray temperatures, liquid compositions, vapor composition,
liquid flow, vapor flow, and K-values; product flowrates; and condenser and reboiler duties.

## Setting up a custom simulation

The easiest starting point is to copy
`examples/three_component_column.yaml`. Select chemicals already present in
`component_database.yaml`, then edit the column, feed, and operating values.

### Custom feed flowrate and mixture

An overall feed is specified with `total_flow_kmol_h`, a normalized mole-fraction
`composition`, and a `vapor_fraction` from 0 (all liquid) to 1 (all vapor):

```yaml
component_database: ../component_database.yaml
components:
  - propane
  - i-butane
  - n-pentane

feeds:
  - name: main_feed
    stage: 5
    temperature_k: 300.0
    pressure_kpa: 191.1
    total_flow_kmol_h: 25.0
    composition:
      propane: 0.20
      i-butane: 0.50
      n-pentane: 0.30
    vapor_fraction: 0.25
```

The composition values are mole fractions and must sum to 1. Named values are
recommended because their meaning does not depend on list order. A positional
list such as `[0.20, 0.50, 0.30]` is also accepted and follows the exact order
under `components`.

If the liquid and vapor portions have different compositions, specify their
component molar flowrates directly instead:

```yaml
feeds:
  - name: two_phase_feed
    stage: 5
    temperature_k: 300.0
    pressure_kpa: 191.1
    liquid_component_flow_kmol_h:
      propane: 1.0
      i-butane: 6.0
      n-pentane: 5.0
    vapor_component_flow_kmol_h:
      propane: 4.0
      i-butane: 2.0
      n-pentane: 0.5
```

These entries are flows, not fractions, so they do not need to sum to 1.
Omitted selected components receive zero flow. Multiple feeds are supported by
adding more entries under `feeds`; feeds sharing a stage must have the same
temperature and pressure.

### Column and operating inputs

A minimal complete column section looks like this:

```yaml
column:
  equilibrium_trays: 8
  condenser_type: total
  reboiler_type: partial
  pressure_profile:
    top_kpa: 180.0
    bottom_kpa: 205.0

reflux_ratios: [2.0]

operating_specification:
  kind: distillate_flow
  value_kmol_h: 10.0
```

The supported operating specifications are `distillate_flow`, `bottoms_flow`,
and `perfect_split`. A fixed product flow must be positive and smaller than the
total feed. `perfect_split` computes the distillate flow from the total feed of
explicitly named light components; it is a flow closure and initial estimate,
not a guarantee that the calculated products are perfectly pure.

Run the copied configuration with:

```text
python -m distillation.cli simulate path/to/custom_column.yaml --output results/custom
```

## Numerical method

The solver performs the following operations each outer iteration:

1. Peng-Robinson fugacity and K-value evaluation.
2. Banded tridiagonal component-material-balance solves.
3. Normalization and VLE iteration for every stage.
4. Bracketed bubble-point temperature solves.
5. Liquid and vapor enthalpy evaluation.
6. Direct lower-bidiagonal internal-flow solution.
7. Convergence and invariant checks.

## Units

- pressure: Pa;
- temperature: K;
- molar flow: kmol/h;
- molar enthalpy: J/mol;
- duty: kJ/h.

## Outputs

Each reflux-ratio run writes:

- `R_<ratio>.json`: complete profiles and diagnostics;
- `R_<ratio>.npz`: compressed numeric arrays;
- `R_<ratio>_stages.csv`: stage table;
- `summary.json`: compact convergence and duty summary;
- `plots/*.png`: liquid composition, vapor composition, temperature, internal
  flows, and duty versus reflux ratio.
