from __future__ import annotations

import numpy as np
import pytest

from distillation.components import (
    Component,
    ComponentSet,
    PropertyDataError,
    load_component_database,
)
from distillation.configuration import ConfigurationError, config_from_mapping, load_config


def test_hydrocarbon_property_parsing_and_units(project_root):
    config = load_config(project_root / "examples" / "hydrocarbon_column.yaml")
    assert len(config.components) == 6
    assert config.components.names[0] == "propane"
    assert config.components.pc[0] == pytest.approx(4.248e6)
    assert config.components.tc[0] == pytest.approx(369.83)
    assert config.n_stages == 32
    assert config.stage_pressures_pa[0] == pytest.approx(config.stage_pressures_pa[1])
    assert config.stage_pressures_pa[-1] == pytest.approx(config.stage_pressures_pa[-2])
    assert [feed.stage for feed in config.feeds] == [13, 17]


def test_shared_component_database_is_public_and_complete(project_root):
    database = load_component_database(project_root / "component_database.yaml")
    assert set(database) >= {
        "propane",
        "n-butane",
        "i-butane",
        "n-pentane",
        "i-pentane",
        "n-hexane",
        "water",
        "ethanol",
        "formic_acid",
    }
    assert database["ethanol"].critical_pressure_pa == pytest.approx(6.137e6)


def test_examples_select_components_from_shared_database(project_root):
    hydrocarbon = load_config(project_root / "examples" / "hydrocarbon_column.yaml")
    small = load_config(project_root / "examples" / "three_component_column.yaml")
    assert hydrocarbon.components.names == (
        "propane",
        "n-butane",
        "i-butane",
        "n-pentane",
        "i-pentane",
        "n-hexane",
    )
    assert small.components.names == ("propane", "i-butane", "n-pentane")


def test_overall_feed_input_and_one_based_stage(project_root):
    config = load_config(project_root / "examples" / "three_component_column.yaml")
    feed = config.feeds[0]
    assert feed.stage == 5
    assert feed.total_flow_kmol_h == pytest.approx(20.0)
    assert feed.vapor_flow_kmol_h == pytest.approx(7.0)
    assert np.allclose(feed.component_flow_kmol_h / feed.total_flow_kmol_h, [0.3, 0.4, 0.3])


def test_named_feed_composition_is_independent_of_mapping_order(project_root):
    config = load_config(project_root / "examples" / "three_component_column.yaml")
    feed = config.feeds[0]
    assert np.allclose(feed.component_flow_kmol_h, [6.0, 8.0, 6.0])


def test_named_phase_component_flows(project_root):
    import yaml

    source = project_root / "examples" / "three_component_column.yaml"
    data = yaml.safe_load(source.read_text())
    database = load_component_database(project_root / "component_database.yaml")
    data["components"] = [database[name] for name in data["components"]]
    data["feeds"][0].pop("total_flow_kmol_h")
    data["feeds"][0].pop("composition")
    data["feeds"][0].pop("vapor_fraction")
    data["feeds"][0]["liquid_component_flow_kmol_h"] = {
        "n-pentane": 5.0,
        "i-butane": 6.0,
    }
    data["feeds"][0]["vapor_component_flow_kmol_h"] = {
        "propane": 4.0,
        "i-butane": 2.0,
    }
    feed = config_from_mapping(data).feeds[0]
    assert np.allclose(feed.liquid_component_flow_kmol_h, [0.0, 6.0, 5.0])
    assert np.allclose(feed.vapor_component_flow_kmol_h, [4.0, 2.0, 0.0])


def test_named_feed_rejects_unknown_component(project_root):
    import yaml

    source = project_root / "examples" / "three_component_column.yaml"
    data = yaml.safe_load(source.read_text())
    database = load_component_database(project_root / "component_database.yaml")
    data["components"] = [database[name] for name in data["components"]]
    data["feeds"][0]["composition"]["methane"] = 0.0
    with pytest.raises(ConfigurationError, match="unknown component.*methane"):
        config_from_mapping(data)


def test_invalid_property_data_is_helpful():
    with pytest.raises(PropertyDataError, match="reference_enthalpy"):
        Component.from_mapping(
            {
                "name": "bad",
                "critical_temperature_k": 300,
                "critical_pressure_pa": 1e6,
                "acentric_factor": 0.1,
                "ideal_gas_cp_coefficients": [1, 2, 3, 4, 5],
            }
        )
    c = Component("a", 300, 1e6, 0.1, (1, 2, 3, 4, 5), 0)
    with pytest.raises(PropertyDataError, match="shape"):
        ComponentSet([c], [[0, 1], [1, 0]])


def test_feed_stage_out_of_range(project_root):
    import yaml

    source = project_root / "examples" / "three_component_column.yaml"
    data = yaml.safe_load(source.read_text())
    database = load_component_database(project_root / "component_database.yaml")
    data["components"] = [database[name] for name in data["components"]]
    data["feeds"][0]["stage"] = 99
    with pytest.raises(ConfigurationError, match="outside engineering stages"):
        config_from_mapping(data)


def test_unknown_database_component_has_helpful_error(project_root, tmp_path):
    import yaml

    source = project_root / "examples" / "three_component_column.yaml"
    data = yaml.safe_load(source.read_text())
    data["component_database"] = str(project_root / "component_database.yaml")
    data["components"] = ["propane", "does-not-exist"]
    path = tmp_path / "bad_column.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="does-not-exist.*available"):
        load_config(path)


def test_duplicate_database_names_are_rejected(tmp_path):
    path = tmp_path / "duplicates.yaml"
    path.write_text(
        """
components:
  - &chemical
    name: duplicate
    critical_temperature_k: 300
    critical_pressure_mpa: 3
    acentric_factor: 0.1
    ideal_gas_cp_coefficients: [1, 2, 3, 4, 5]
    reference_enthalpy_j_mol: 0
  - <<: *chemical
""",
        encoding="utf-8",
    )
    with pytest.raises(PropertyDataError, match="unique"):
        load_component_database(path)
