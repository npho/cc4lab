import jax
import jax.numpy as jnp
import pytest
from jaxborg.checkpoint import save_jax_params, write_sidecar
from jaxborg.policies import make_jax_policy

from cc4lab.policies.loaders import checkpoint_info, load_policy

N, OBS, ACT = 5, 210, 242
ARCH = {"name": "shared", "hidden_dim": 32, "hidden_layers": 2, "activation": "tanh"}


def _write_checkpoint(tmp_path, arch=ARCH):
    net = make_jax_policy(
        arch["name"], action_dim=ACT, hidden_dim=arch["hidden_dim"],
        hidden_layers=arch["hidden_layers"], activation=arch["activation"],
    )  # fmt: skip
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS)), jnp.ones((1, ACT)))
    model = tmp_path / "model_t.safetensors"
    save_jax_params(model, params, action_dim=ACT)
    recipe = {"meta": {"name": "toy"}, "algorithm": "ippo", "arch": arch}
    write_sidecar(tmp_path / "recipe_t.yaml", recipe, seed=7, total_steps=123, backend="jax")
    return model, net, params


def _inputs():
    obs = jax.random.normal(jax.random.PRNGKey(1), (N, OBS))
    masks = jax.random.bernoulli(jax.random.PRNGKey(2), 0.3, (N, ACT)).at[:, 0].set(True)
    keys = jax.random.split(jax.random.PRNGKey(3), N)
    return keys, obs, masks


def test_policy_matches_manual_apply_and_respects_masks(tmp_path):
    model, net, params = _write_checkpoint(tmp_path)
    policy = load_policy(model)
    keys, obs, masks = _inputs()

    actions = policy(keys, obs, masks)
    assert actions.shape == (N,) and actions.dtype == jnp.int32
    assert bool(masks[jnp.arange(N), actions].all())  # only valid actions

    pi, _ = net.apply(params, obs, masks.astype(jnp.float32))
    expected = jnp.stack([jax.random.categorical(keys[i], pi.logits[i]) for i in range(N)])
    assert jnp.array_equal(actions, expected)


def test_each_agent_uses_only_its_own_key(tmp_path):
    model, _, _ = _write_checkpoint(tmp_path)
    policy = load_policy(model)
    keys, obs, masks = _inputs()
    other = keys.at[2].set(jax.random.PRNGKey(99))
    a, b = policy(keys, obs, masks), policy(other, obs, masks)
    assert jnp.array_equal(jnp.delete(a, 2), jnp.delete(b, 2))  # other agents unaffected


def test_checkpoint_info_reports_provenance(tmp_path):
    model, _, _ = _write_checkpoint(tmp_path)
    info = checkpoint_info(model)
    assert info["recipe"] == "toy" and info["seed"] == 7 and info["total_steps"] == 123


def test_unsupported_arch_is_rejected(tmp_path):
    model, _, _ = _write_checkpoint(tmp_path)
    side = tmp_path / "recipe_t.yaml"
    side.write_text(side.read_text().replace("name: shared", "name: per_agent"))
    with pytest.raises(NotImplementedError):
        load_policy(model)
