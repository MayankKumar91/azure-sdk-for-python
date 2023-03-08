# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import yaml
import pytest

from azure.ai.ml.entities._assets._artifacts.feature_set import FeatureSet
from azure.ai.ml import load_feature_set


@pytest.mark.unittest
@pytest.mark.data_experiences_test
class TestFeatureSetSchema:
    def test_feature_set_load(self) -> None:
        test_path = "./tests/test_configs/feature_set/feature_set_full.yaml"
        with open(test_path, "r") as f:
            target = yaml.safe_load(f)
        with open(test_path, "r") as f:
            feature_set: FeatureSet = load_feature_set(source=test_path)
        assert feature_set.name == target["name"]
        assert feature_set.version == target["version"]
        assert feature_set.description == target["description"]
        assert feature_set.entities is not None
        assert feature_set.specification is not None
        assert feature_set.specification.path is not None
        assert feature_set.tags is not None
        assert feature_set.properties is not None
