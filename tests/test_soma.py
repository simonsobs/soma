import somapy


def test_version_is_exposed():
    assert isinstance(somapy.__version__, str)
    assert somapy.__version__


def test_maps_module_imports():
    from somapy import maps

    assert maps is not None
