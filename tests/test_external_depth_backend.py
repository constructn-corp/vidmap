"""The depth stage's external backend: config, cache identity and dispatch."""

import sys
import types

import pytest

from vidmap.configuration.config import FrontendConfig
from vidmap.configuration.dump import config_to_dict
from vidmap.frontend import depth
from vidmap.frontend.options.depth import DepthEstimationOptions


def _backend_module(monkeypatch, name="fake_depth_backend", **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_unset_external_backend_keeps_the_da3_cache_token_and_keys():
    options = DepthEstimationOptions()
    assert options.external_backend is None
    assert options.depth_model == "da3_video_w1"
    assert "external_backend" not in depth._depth_options_cache_config(options)
    assert "external_backend" not in config_to_dict(options)


def test_external_backend_has_its_own_cache_token_and_keys():
    options = DepthEstimationOptions(external_backend="pkg.pano_depth")
    assert options.depth_model == "external_pkg_pano_depth"
    assert depth._depth_options_cache_config(options)["external_backend"] == "pkg.pano_depth"
    assert config_to_dict(options)["external_backend"] == "pkg.pano_depth"


@pytest.mark.parametrize("name", ["", "../adapter", "pano-depth", "pkg..mod", "1mod", "mod.py/x"])
def test_external_backend_must_be_a_module_name(name):
    with pytest.raises(ValueError, match="dotted Python module name"):
        DepthEstimationOptions(external_backend=name)


def test_frontend_config_accepts_the_external_backend_from_a_mapping():
    config = FrontendConfig(depth={"external_backend": "pano_adapter", "backend": {"compile": False}})
    assert config.depth.external_backend == "pano_adapter"
    assert config.depth.backend.compile is False


def test_unset_external_backend_dispatches_to_da3(monkeypatch):
    calls = []
    monkeypatch.setattr(depth, "_run_da3_depth", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(depth, "da3_cache_identity", lambda backend: {"da3": backend.window_size})
    options = DepthEstimationOptions()
    depth.run_depth_backend(options, "scene", ["a"], ["a"], device="cpu")
    assert calls == [(("scene", ["a"], ["a"]), {"device": "cpu"})]
    assert depth.depth_backend_identity(options) == {"da3": 1}


def test_external_backend_receives_the_da3_arguments_and_names_the_cache(monkeypatch):
    calls = []
    _backend_module(
        monkeypatch,
        cache_identity=lambda options: {"model": "pano", "batch_size": options.batch_size},
        run_depth=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(depth, "_run_da3_depth", lambda *args, **kwargs: pytest.fail("DA3 must not run"))
    options = DepthEstimationOptions(external_backend="fake_depth_backend", batch_size=2)
    depth.run_depth_backend(options, "scene", ["a"], ["a"], device="cpu")
    assert calls == [(("scene", ["a"], ["a"]), {"device": "cpu"})]
    identity = depth.depth_backend_identity(options)
    assert identity["external_backend"] == "fake_depth_backend"
    assert identity["identity"] == {"model": "pano", "batch_size": 2}


def test_external_backend_without_the_api_is_rejected(monkeypatch):
    _backend_module(monkeypatch, name="incomplete_depth_backend", cache_identity=lambda options: {})
    with pytest.raises(TypeError, match="does not define run_depth"):
        depth.external_depth_backend("incomplete_depth_backend")


def test_missing_external_backend_module_fails_on_import():
    with pytest.raises(ModuleNotFoundError):
        depth.external_depth_backend("vidmap_no_such_depth_backend")
