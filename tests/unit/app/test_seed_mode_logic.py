from __future__ import annotations

from pmarlo_webapp.app.backend import choose_sim_seed


def test_choose_sim_seed_modes():
    assert choose_sim_seed("none") is None
    assert choose_sim_seed("fixed", fixed=7) == 7
    a = choose_sim_seed("auto")
    b = choose_sim_seed("auto")
    assert isinstance(a, int) and isinstance(b, int)
    assert a != b
