import jax

from cc4lab import env as E


def test_describe_backend_reports_device_and_commit():
    info = E.describe_backend(print_it=False)
    assert info["devices"]
    assert len(info["jaxborg_commit"].split("+")[0]) == 40


def test_reset_shapes():
    env = E.make_env()
    obs, state = E.reset(env, jax.random.PRNGKey(0))
    assert obs.shape == (E.N_BLUE, E.OBS_DIM) == (5, 210)
    assert E.action_masks(env, state).shape == (E.N_BLUE, E.ACTION_DIM) == (5, 242)
    assert int(state.state.time) == 0
