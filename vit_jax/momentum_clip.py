
import flax
import jax
import jax.numpy as jnp
import numpy as np


class Optimizer(flax.optim.OptimizerDef):
  """Momentum optimizer(반정밀도를 이용한 state 저장)"""

  @flax.struct.dataclass # 이 클래스의 instance들을 jax로 전달할 수 있도록 함.
  class HyperParams:
    learning_rate: np.ndarray
    beta: np.ndarray
    grad_norm_clip: np.ndarray

  @flax.struct.dataclass
  class State:
    momentum: np.ndarray

  def __init__(self,
               learning_rate=None,
               beta=0.9,
               dtype='bfloat16',
               grad_norm_clip=None):
    hyper_params = Optimizer.HyperParams(learning_rate, beta, grad_norm_clip)
    super().__init__(hyper_params)
    self.dtype = dict(bfloat16=jnp.bfloat16, float32=jnp.float32)[dtype]

  def init_param_state(self, param):
    return Optimizer.State(jnp.zeros_like(param, dtype=self.dtype)) # momentum:0으로 초기화

  def apply_gradient(self, hyper_params, params, state, grads):
    step = state.step
    params_flat, treedef = jax.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)

    # Optionally resize the global gradient to a maximum norm.
    if hyper_params.grad_norm_clip:
      grads_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
      grads_factor = jnp.minimum(1.0, hyper_params.grad_norm_clip / grads_l2)
      grads_flat = jax.tree_map(lambda param: grads_factor * param, grads_flat)

    out = [
        self.apply_param_gradient(step, hyper_params, param, state, grad)
        for param, state, grad in zip(params_flat, states_flat, grads_flat)
    ]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = flax.optim.OptimizerState(step + 1, new_param_states)
    return new_params, new_state

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step # 지움
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + grad # 모멘텀 방정식
    new_param = param - hyper_params.learning_rate * new_momentum
    new_state = Optimizer.State(new_momentum.astype(self.dtype))
    return new_param, new_state
