
import flax
import jax
import numpy as np


def create_learning_rate_schedule(total_steps,
                                  base,
                                  decay_type,
                                  warmup_steps,
                                  linear_end=1e-5):
  """
      warmup + {linear, cosine}.

  Args:
    total_steps: run해야하는 step의 총 갯수
    base: 시작 learning rate (without warmup)
    decay_type: 'linear' 또는 'cosine'
    warmup_steps: warm up하기 위해 필요한 스텝 수
    linear_end: 최소 learning rate

  Returns:
    함수 learning_rate(step): float -> {"learning_rate": float}
  """

  def step_fn(step):
    """learning rate fn을 위한 step."""
    lr = base # 시작 lr

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = np.clip(progress, 0.0, 1.0) # 음수면 0, 1보다 크면 1로 조정
    if decay_type == 'linear':
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
      lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    else: # decay_type 오류
      raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
      lr = lr * np.minimum(1., step / warmup_steps) # 작은 값부터 점점 1에 가까운 수 곱함

    return np.asarray(lr, dtype=np.float32) # lr 반환

  return step_fn


def lr_prefetch_iter(lr_fn,
                     first_step,
                     total_steps,
                     prefetch_to_device=2,
                     devices=None):
  local_device_count = (
      jax.local_device_count() if devices is None else len(devices))
  lr_iter = (
      np.ones([local_device_count]) * lr_fn(i)
      for i in range(first_step, total_steps))
  # TPU 전송 오버헤드를 줄이기 위해 lr을 prefetching.
  return flax.jax_utils.prefetch_to_device(
      lr_iter, prefetch_to_device, devices=devices)


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
  """step마다 gradient 누적해서 메모리에 저장"""
  if accum_steps and accum_steps > 1: # 0이상
    assert images.shape[0] % accum_steps == 0, (
        f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
    step_size = images.shape[0] // accum_steps
    l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

    def acc_grad_and_loss(i, l_and_g): # 정확도
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      li, gi = loss_and_grad_fn(params, imgs, lbls)
      l, g = l_and_g
      return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree_map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)
