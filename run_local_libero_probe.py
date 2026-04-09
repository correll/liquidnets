import argparse
import os
import sys
from pathlib import Path

import numpy as np

TASK_PRESETS = {
    'drawer_close': 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet',
    'turn_on_stove': 'turn_on_the_stove',
    'push_plate': 'push_the_plate_to_the_front_of_the_stove',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal LIBERO local eval/probe runner.')
    parser.add_argument(
        '--task',
        choices=sorted(TASK_PRESETS.keys()),
        default='drawer_close',
        help='Task preset to evaluate.',
    )
    parser.add_argument(
        '--task-stem',
        type=str,
        default='',
        help='Optional explicit BDDL stem override (without .bddl).',
    )
    parser.add_argument('--steps', type=int, default=50, help='Max rollout steps.')
    parser.add_argument('--camera-size', type=int, default=128, help='Render resolution (HxW).')
    return parser.parse_args()


def add_local_libero_repo_to_path():
    repo_candidates = [Path.cwd() / '_tmp_LIBERO', Path.cwd() / 'LIBERO']
    repo_root = next((p for p in repo_candidates if p.exists()), None)
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
    return repo_root


def configure_libero_paths(repo_root: Path | None):
    cfg_dir = Path.cwd() / '.libero_local'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg_dir)

    if repo_root is not None:
        benchmark_root = repo_root / 'libero' / 'libero'
    else:
        benchmark_root = Path('/usr/local/lib/python3.12/dist-packages/libero/libero')

    cfg_file = cfg_dir / 'config.yaml'
    cfg_text = (
        f"benchmark_root: {benchmark_root}\n"
        f"bddl_files: {benchmark_root / 'bddl_files'}\n"
        f"init_states: {benchmark_root / 'init_files'}\n"
        f"datasets: {benchmark_root.parent / 'datasets'}\n"
        f"assets: {benchmark_root / 'assets'}\n"
    )
    cfg_file.write_text(cfg_text, encoding='utf-8')
    return cfg_file


def resolve_bddl_file(task_stem: str):
    candidates = []
    try:
        import libero.libero as libero_core

        candidates.append(Path(libero_core.get_libero_path('bddl_files')))
    except Exception:
        pass

    candidates.extend([
        Path.cwd() / '_tmp_LIBERO' / 'libero' / 'libero' / 'bddl_files',
        Path.cwd() / 'libero' / 'libero' / 'bddl_files',
        Path('/usr/local/lib/python3.12/dist-packages/libero/libero/bddl_files'),
    ])

    for root in candidates:
        if root.exists():
            found = sorted(root.rglob(task_stem + '.bddl'))
            if found:
                return found[0]
    return None


def unwrap_env_candidates(root):
    attrs = [
        'env', '_env', 'unwrapped', 'wrapped_env', '_wrapped_env',
        'venv', 'gym_env', 'base_env', '_inner_env', 'inner_env',
    ]
    stack, seen = [root], set()
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for a in attrs:
            try:
                nxt = getattr(cur, a, None)
            except Exception:
                nxt = None
            if nxt is not None and id(nxt) not in seen:
                stack.append(nxt)


def get_success_signal(candidates):
    method_names = [
        '_check_success', 'check_success', 'is_success', 'success',
        '_check_task_success', '_is_success', 'task_success',
        'get_success', 'get_task_success', '_get_success',
    ]
    for c in candidates:
        for n in method_names:
            fn = getattr(c, n, None)
            if callable(fn):
                try:
                    v = fn()
                    if isinstance(v, (bool, np.bool_, int, float)):
                        return float(bool(v)), f'{type(c).__name__}.{n}'
                except Exception:
                    pass
    return 0.0, None


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, info if isinstance(info, dict) else {}


def main():
    args = parse_args()
    task_stem = args.task_stem.strip() or TASK_PRESETS[args.task]

    print('task preset :', args.task)
    print('task stem   :', task_stem)

    repo_root = add_local_libero_repo_to_path()
    print('repo_root   :', repo_root)

    cfg_file = configure_libero_paths(repo_root)
    print('config file :', cfg_file)

    bddl_file = resolve_bddl_file(task_stem)
    if bddl_file is None:
        raise FileNotFoundError(f'Could not find BDDL for task stem: {task_stem}')
    print('bddl file   :', bddl_file)

    try:
        text = Path(bddl_file).read_text(encoding='utf-8', errors='ignore')
        i = text.find('(:goal')
        print('goal snippet:', text[i:i + 220] if i >= 0 else 'No (:goal ...) section found')
    except Exception:
        pass

    try:
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as e:
        raise RuntimeError(f'LIBERO import failed: {e}') from e

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(args.camera_size),
        camera_widths=int(args.camera_size),
    )

    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        candidates = list(unwrap_env_candidates(env))

        ep_return = 0.0
        ep_success = 0.0
        success_source = None

        for step in range(int(args.steps)):
            action = np.zeros(7, dtype=np.float32)
            obs, reward, terminated, truncated, info = safe_step(env, action)
            ep_return += reward

            s, src = get_success_signal(candidates)
            if success_source is None and src is not None:
                success_source = src
            ep_success = max(ep_success, s)

            done = bool(terminated or truncated)
            if step % 10 == 0 or done:
                print(
                    f'step={step:03d} reward={reward:.3f} done={done} '
                    f'success={ep_success:.1f} src={success_source} '
                    f'info_keys={list(info.keys()) if isinstance(info, dict) else []}'
                )
            if done:
                break

        print('\nsummary')
        print('  return         :', round(ep_return, 4))
        print('  success        :', float(ep_success))
        print('  success_source :', success_source)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
