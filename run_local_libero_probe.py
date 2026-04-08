from pathlib import Path
import os
import sys
import numpy as np

TASK_STEM = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet'
print('task stem:', TASK_STEM)

repo_candidates = [Path.cwd() / '_tmp_LIBERO', Path.cwd() / 'LIBERO']
repo_root = next((p for p in repo_candidates if p.exists()), None)
if repo_root is not None:
    sys.path.insert(0, str(repo_root))
print('repo_root:', repo_root)

libero_cfg_dir = Path.cwd() / '.libero_local'
libero_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ['LIBERO_CONFIG_PATH'] = str(libero_cfg_dir)

if repo_root is not None:
    benchmark_root = repo_root / 'libero' / 'libero'
else:
    benchmark_root = Path('/usr/local/lib/python3.12/dist-packages/libero/libero')

cfg_file = libero_cfg_dir / 'config.yaml'
cfg_text = (
    f"benchmark_root: {benchmark_root}\n"
    f"bddl_files: {benchmark_root / 'bddl_files'}\n"
    f"init_states: {benchmark_root / 'init_files'}\n"
    f"datasets: {benchmark_root.parent / 'datasets'}\n"
    f"assets: {benchmark_root / 'assets'}\n"
)
cfg_file.write_text(cfg_text, encoding='utf-8')
print('config file:', cfg_file)


def resolve_bddl_file(task_stem: str):
    candidates = []
    try:
        import libero.libero as libero_core

        bddl_root = Path(libero_core.get_libero_path('bddl_files'))
        candidates.append(bddl_root)
    except Exception as e:
        print('[warn] libero config lookup failed:', e)

    candidates.extend([
        Path.cwd() / '_tmp_LIBERO' / 'libero' / 'libero' / 'bddl_files',
        Path.cwd() / 'libero' / 'libero' / 'bddl_files',
        Path('/usr/local/lib/python3.12/dist-packages/libero/libero/bddl_files'),
    ])
    for root in candidates:
        if not root.exists():
            continue
        found = sorted(root.rglob(task_stem + '.bddl'))
        if found:
            return found[0]
    return None


bddl_file = resolve_bddl_file(TASK_STEM)
print('bddl_file =', bddl_file)
if bddl_file:
    txt = Path(bddl_file).read_text(encoding='utf-8', errors='ignore')
    i = txt.find('(:goal')
    print('goal snippet:')
    print(txt[i : i + 280] if i >= 0 else 'No (:goal found)')

env = None


def unwrap_env_candidates(root):
    attrs = [
        'env', '_env', 'unwrapped', 'wrapped_env', '_wrapped_env',
        'venv', 'gym_env', 'base_env', '_inner_env', 'inner_env'
    ]
    stack, seen = [root], set()
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        yield cur
        for a in attrs:
            try:
                nxt = getattr(cur, a, None)
            except Exception:
                nxt = None
            if nxt is not None and id(nxt) not in seen:
                stack.append(nxt)


try:
    from libero.libero.envs import OffScreenRenderEnv

    if bddl_file is not None:
        for kwargs in [
            {'bddl_file_name': str(bddl_file), 'camera_heights': 128, 'camera_widths': 128},
            {'bddl_file_name': str(bddl_file)},
        ]:
            try:
                env = OffScreenRenderEnv(**kwargs)
                print('Env built with kwargs:', kwargs)
                break
            except Exception as e:
                print('Env build failed:', e)
except Exception as e:
    print('LIBERO import/build failure:', e)

if env is not None:
    cands = list(unwrap_env_candidates(env))
    print('wrapper count:', len(cands))
    for i, c in enumerate(cands):
        sattrs = [n for n in dir(c) if 'success' in n.lower()]
        print(f'[{i}] {type(c).__module__}.{type(c).__name__}')
        if sattrs:
            print('   success-like attrs:', sattrs[:20])


def call_success_hooks(cands):
    out = []
    names = [
        '_check_success', 'check_success', 'is_success', 'success',
        '_check_task_success', '_is_success', 'task_success',
        'get_success', 'get_task_success', '_get_success'
    ]
    for c in cands:
        for n in names:
            fn = getattr(c, n, None)
            if callable(fn):
                try:
                    v = fn()
                    out.append((type(c).__name__, n, v))
                except TypeError:
                    pass
                except Exception as e:
                    out.append((type(c).__name__, n, f'ERR:{type(e).__name__}:{e}'))
    return out


if env is None:
    print('No env. Stop.')
else:
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
        info = reset_out[1] if len(reset_out) > 1 and isinstance(reset_out[1], dict) else {}
    else:
        obs, info = reset_out, {}

    print('reset info keys:', list(info.keys()) if isinstance(info, dict) else type(info))
    cands = list(unwrap_env_candidates(env))

    for step in range(20):
        a = np.zeros(7, dtype=np.float32)
        out = env.step(a)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False

        hooks = call_success_hooks(cands)
        hit = [
            (c, n, v)
            for (c, n, v) in hooks
            if isinstance(v, (bool, np.bool_, int, float)) and bool(v)
        ]
        print(
            f"step={step:02d} reward={float(reward):.3f} done={done} "
            f"term={terminated} trunc={truncated} "
            f"info_keys={list(info.keys()) if isinstance(info, dict) else type(info)}"
        )
        if hit:
            print('  TRUE success hooks:', hit[:5])
        if done:
            print('Episode ended.')
            break

    try:
        env.close()
    except Exception:
        pass
