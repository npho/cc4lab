import jax

from cc4lab import env as E

SHA = "b85bc324851808708a597b2bba3a84994ffff7d9"


def test_describe_backend_reports_device_and_commit():
    info = E.describe_backend(print_it=False)
    assert info["devices"]
    assert len(info["jaxborg_commit"].split("+")[0]) == 40


def test_commit_from_git_install_metadata():
    info = {"url": "https://github.com/ITM-Kitware/jaxborg", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    assert E._commit_from_direct_url(info) == SHA


def test_commit_from_local_checkout_tracks_dirty(tmp_path):
    import subprocess

    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    info = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    clean = E._commit_from_direct_url(info)
    assert len(clean) == 40 and "+" not in clean
    (tmp_path / "f").write_text("x")
    assert E._commit_from_direct_url(info) == clean + "+dirty"


def test_commit_unknown_without_metadata():
    assert E._commit_from_direct_url({}) == "unknown"


def test_reset_shapes():
    env = E.make_env()
    obs, state = E.reset(env, jax.random.PRNGKey(0))
    assert obs.shape == (E.N_BLUE, E.OBS_DIM) == (5, 210)
    assert E.action_masks(env, state).shape == (E.N_BLUE, E.ACTION_DIM) == (5, 242)
    assert int(state.state.time) == 0
