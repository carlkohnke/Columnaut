"""Dense-array column model independent of thermodynamic-package internals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .configuration import ColumnConfig, ConfigurationError


@dataclass(slots=True)
class ColumnModel:
    config: ColumnConfig
    feed_total: NDArray[np.float64]
    feed_liquid: NDArray[np.float64]
    feed_vapor: NDArray[np.float64]
    feed_component: NDArray[np.float64]
    liquid_feed_composition: NDArray[np.float64]
    vapor_feed_composition: NDArray[np.float64]
    overall_feed_composition: NDArray[np.float64]
    feed_temperature_k: NDArray[np.float64]
    feed_pressure_pa: NDArray[np.float64]

    @classmethod
    def from_config(cls, config: ColumnConfig) -> "ColumnModel":
        ns = config.n_stages
        nc = len(config.components)
        lf = np.zeros(ns)
        vf = np.zeros(ns)
        fc = np.zeros((ns, nc))
        lcomp = np.zeros((ns, nc))
        vcomp = np.zeros((ns, nc))
        z = np.zeros((ns, nc))
        ft = np.full(ns, np.nan)
        fp = np.full(ns, np.nan)
        for feed in config.feeds:
            s = feed.stage - 1
            # Multiple feeds on one engineering stage are combined by flow.
            previous_l = lf[s]
            previous_v = vf[s]
            liquid_vector = lcomp[s] * previous_l + feed.liquid_component_flow_kmol_h
            vapor_vector = vcomp[s] * previous_v + feed.vapor_component_flow_kmol_h
            lf[s] += feed.liquid_flow_kmol_h
            vf[s] += feed.vapor_flow_kmol_h
            fc[s] += feed.component_flow_kmol_h
            if lf[s] > 0:
                lcomp[s] = liquid_vector / lf[s]
            if vf[s] > 0:
                vcomp[s] = vapor_vector / vf[s]
            total = lf[s] + vf[s]
            z[s] = fc[s] / total
            if np.isnan(ft[s]):
                ft[s] = feed.temperature_k
                fp[s] = feed.pressure_pa
            elif not (
                np.isclose(ft[s], feed.temperature_k, atol=1e-12)
                and np.isclose(fp[s], feed.pressure_pa, atol=1e-6)
            ):
                raise ConfigurationError(
                    f"Multiple feeds on stage {feed.stage} must share temperature and pressure"
                )
        return cls(config, lf + vf, lf, vf, fc, lcomp, vcomp, z, ft, fp)

    @property
    def total_component_feed(self) -> NDArray[np.float64]:
        return self.feed_component.sum(axis=0)

    @property
    def total_feed(self) -> float:
        return float(self.feed_total.sum())

    def product_flows(self) -> tuple[float, float, tuple[int, ...]]:
        spec = self.config.operating_specification
        light = self.config.components.indices(spec.light_components) if spec.light_components else ()
        if spec.kind == "perfect_split":
            if not light:
                raise ConfigurationError("perfect_split specification requires light_components")
            distillate = float(self.total_component_feed[list(light)].sum())
        elif spec.kind == "distillate_flow":
            if spec.value_kmol_h is None:
                raise ConfigurationError("distillate_flow specification requires value_kmol_h")
            distillate = spec.value_kmol_h
        elif spec.kind == "bottoms_flow":
            if spec.value_kmol_h is None:
                raise ConfigurationError("bottoms_flow specification requires value_kmol_h")
            distillate = self.total_feed - spec.value_kmol_h
        else:
            raise ConfigurationError(
                f"Unsupported operating specification {spec.kind!r}; use perfect_split, distillate_flow, or bottoms_flow"
            )
        bottoms = self.total_feed - distillate
        if distillate <= 0 or bottoms <= 0:
            raise ConfigurationError(
                f"Product specification gives distillate={distillate:g}, bottoms={bottoms:g} kmol/h"
            )
        return float(distillate), float(bottoms), light
