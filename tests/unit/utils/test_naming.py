import pmarlo.utils.naming as naming


def test_base_shape_str_cache():
    naming.base_shape_str.cache_clear()
    naming.base_shape_str((2, 3))
    assert naming.base_shape_str.cache_info().hits == 0
    naming.base_shape_str((2, 3))
    assert naming.base_shape_str.cache_info().hits == 1
    assert naming.base_shape_str((2, 3)) == "2x3"


def test_permutation_name_cache():
    naming.permutation_name.cache_clear()
    perm = (1, 0, 2)
    naming.permutation_name(perm)
    assert naming.permutation_name.cache_info().hits == 0
    naming.permutation_name(perm)
    assert naming.permutation_name.cache_info().hits == 1
    assert naming.permutation_name(perm) == "1-0-2"
