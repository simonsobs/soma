import soma


def test_version_is_exposed():
    assert isinstance(soma.__version__, str)
    assert soma.__version__


def test_maps_module_imports():
    from soma import maps

    assert maps is not None
