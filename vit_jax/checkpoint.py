# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import collections
import dataclasses
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tensorflow.io import gfile  # pylint: disable=import-error


def _flatten_dict(d, parent_key='', sep='/'):
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(_flatten_dict(v, path, sep=sep).items())
    else:
      items.append((path, v))
      
  # 빈 딕셔너리가 명시적으로 설정되어 있으면 유지
  if parent_key and not d:
    items.append((parent_key, {}))

  return dict(items)


def inspect_params(*,
                   params,
                   expected,
                   logger,
                   fail_if_extra=True,
                   fail_if_missing=True):
  """매개변수가 예상 키와 일치하는지 검사"""
  params_flat = _flatten_dict(params)
  expected_flat = _flatten_dict(expected)
  missing_keys = expected_flat.keys() - params_flat.keys()
  extra_keys = params_flat.keys() - expected_flat.keys()

  # 가중치 없는 계층을 지원하기 위해 뒤에 빈 딕셔너리 명시적으로 추가
  # Context: FLAX는 직렬화 과정에서 빈 딕셔너리 무시
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      params[k] = {}
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    logger.warning('Inspect recovered empty keys:\n%s', empty_keys)
  if missing_keys:
    logger.info('Inspect missing keys:\n%s', missing_keys)
  if extra_keys:
    logger.info('Inspect extra keys:\n%s', extra_keys)

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(f'Missing params from checkpoint: {missing_keys}.\n'
                     f'Extra params in checkpoint: {extra_keys}.\n'
                     f'Restored params from checkpoint: {params_flat.keys()}.\n'
                     f'Expected params from code: {expected_flat.keys()}.')
  return params


def recover_tree(keys, values):
  """트리를 1차원 이름과 값에서 중첩된 딕셔너리로 복구한다.

  이 함수는 정확한 소스코드에 액세스할 필요 없는 체크포인트를 분석하는 것에 유용하다.
  특히 체크포인트의 다양한 하위트리(예: 매개변수의 하위트리)를 재사용할 수 있다.

  Args:
    keys: 키들로 구성된 리스트, '/'는 노드 사이 분리기로 사용된다.
    values: 최하위 노드(잎)의 값으로 구성된 리스트.

  Returns:
    중첩된 트리형의 딕셔너리
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def _traverse_with_names(tree):
  """중첩된 딕셔너리/데이터클래스들을 통과하고 (leaf_name, leaf_val)을 내보낸다."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, dict):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + '/' + path).rstrip('/'), v
  else:
    yield '', tree


def tree_flatten_with_names(tree):
  """tree_flatten을 leaf 이름으로 채운다.

  이 함수는 tree_flatten의 출력을 leaf 이름으로 채운다. 이름을 생성하는 사용자 지정 순회가 제공된다. 
  jax' 및 사용자 지정 순회를 자동으로 정렬하기 때문에 사용자 지정 순회는 jax와 같은 순서로 트리를 순회할 필요가 없다.

  Args:
    tree: python tree.

  Returns:
    이름과 값으로 이루어진 리스트: [(name, value), ...]
  """
  vals, tree_def = jax.tree_flatten(tree)

  # jax 내부 트리 순회를 추적하고 이 트리 순회와 호환되도록 사용자 지정 트리 순회를 조정하는 데에 사용되는 "Fake" 토큰 트리
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)
  
  # 사용자 지정 순회는 동일한 수의 leaf을 방문해야 한다.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def save(data, path):
  """체크포인트 작성에 사용: jax pytree 개체를 디스크에 저장합니다.

  이러한 체크포인트는 나중에 `load()`를 사용하여 복구할 수 있다.
  
  Args:
    data: 저장할 임의의 jax pytree
    path: 데이터를 저장할 경로
  """
  names_and_vals, _ = tree_flatten_with_names(data)
  io_buffer = io.BytesIO()

  # savez는 cns에서 제공하지 않는 `seek()` API를 쓴다.
  # 따라서 먼저 체크포인트를 temp 버퍼에 쓰고난 다음 디스크에 쓴다.
  np.savez(io_buffer, **{k: v for k, v in names_and_vals})

  # 인터럽트에 대응을 하기 위해 먼저 체크포인트를 임시 파일에 저장한 다음 실제 경로로 이동시킨다.
  path_tmp = path + '-TEMPORARY'
  gfile.makedirs(os.path.dirname(path))
  with gfile.GFile(path_tmp, 'wb') as f:
    f.write(io_buffer.getvalue())
  gfile.rename(path_tmp, path, overwrite=True)


def load(path):
  """`save()`로 먼저 저장해둔 체크포인트로부터 파라미터를 가져온다."""
  with gfile.GFile(path, 'rb') as f:
    ckpt_dict = np.load(f, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
  return recover_tree(keys, values)


def load_pretrained(*, pretrained_path, init_params, model_config, logger):
  """fine tuning을 위해 미리 학습된 체크포인트를 가져오고 변환한다.
  
  Args:
    pretrained_path: 미리 학습된 체크포인트를 가리키는 파일
    init_params: 모델의 파라미터. 모델의 헤드로 사용되고 그 모델이 저장된 체크포인트와 호환되는지 확인한다.
    model_config: 모델의 구성. 헤드를 구성하고 위치 임베딩의 크기를 조정하는 데 사용한다.
    logger: 진단 메시지 출력에 사용할 logger
    
  Returns:
    'init_params'와 같은 매개변수지만, 'pretrained_path'에서 미리 학습된 가중치를 로드하고 그에 따라 조정된다.
  """

  restored_params = inspect_params(
      params=load(pretrained_path),
      expected=init_params,
      logger=logger,
      fail_if_extra=False,
      fail_if_missing=False)
  
  # 다음은 fine-tuning 작업에서 `representation_size` 값에 따라 fine-tuning 헤드 변형을 구현할 수 있도록 한다.
  # - `None` : drop the whole head and attach a nn.Linear.
  # - same number as in pre-training means : keep the head but reset the last
  #    layer (logits) for the new task.
  if model_config.representation_size is None:
    if 'pre_logits' in restored_params:
      logger.info('load_pretrained: drop-head variant')
      restored_params['pre_logits'] = {}
  restored_params['head']['kernel'] = init_params['head']['kernel']
  restored_params['head']['bias'] = init_params['head']['bias']

  if 'posembed_input' in restored_params.get('Transformer', {}):
    # 위치 임베딩의 그리드를 재조정한다. 파라미터 모양은 (1,N,1024)
    posemb = restored_params['Transformer']['posembed_input']['pos_embedding']
    posemb_new = init_params['Transformer']['posembed_input']['pos_embedding']
    if posemb.shape != posemb_new.shape:
      logger.info('load_pretrained: resized variant: %s to %s', posemb.shape,
                  posemb_new.shape)
      ntok_new = posemb_new.shape[1]

      if model_config.classifier == 'token':
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
      else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

      gs_old = int(np.sqrt(len(posemb_grid)))
      gs_new = int(np.sqrt(ntok_new))
      logger.info('load_pretrained: grid-size from %s to %s', gs_old, gs_new)
      posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

      zoom = (gs_new / gs_old, gs_new / gs_old, 1)
      posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
      posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
      posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
      restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb

  return restored_params
