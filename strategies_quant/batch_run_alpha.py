#!/usr/bin/env python3
"""
批量回测 alpha_*.py 和 v*.py 策略
这些文件不继承 BaseStrategy, 而是自带回测逻辑
"""
import os, sys, subprocess, time, re

sys.stdout.reconfigure(line_buffering=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# Skip list
SKIP = {'var', 'visual', 'alpha_factor_cache', 'backtest_factors'}

# 这些文件有自己的 main() 可以直接跑
RUNNABLE = []
for f in sorted(os.listdir(BASE)):
    if not f.endswith('.py'):
        continue
    name = f.replace('.py', '')
    if name in SKIP:
        continue
    # Match alpha_*.py and v*.py (v followed by digit)
    is_alpha = name.startswith('alpha_')
    is_v = name.startswith('v') and len(name) > 1 and name[1].isdigit()
    if not (is_alpha or is_v):
        continue
    # Check if it has a main block
    fp = os.path.join(BASE, f)
    with open(fp) as fh:
        content = fh.read()
    if '__main__' in content or 'def main(' in content:
            RUNNABLE.append(name)

print(f"发现 {len(RUNNABLE)} 个可运行的 alpha/v 脚本")
print("=" * 80)

results = []

for i, name in enumerate(RUNNABLE):
    print(f"[{i+1}/{len(RUNNABLE)}] {name}...", end=" ", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, '-u', f'{name}.py'],
            capture_output=True, text=True, timeout=600,
            cwd=BASE
        )
        elapsed = time.time() - t0
        out = proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout
        err = proc.stderr[-300:] if len(proc.stderr) > 300 else proc.stderr

        # Try to extract key metrics from output
        sharpe = ''
        ret = ''
        for line in (proc.stdout + proc.stderr).split('\n'):
            if 'sharpe' in line.lower() and not sharpe:
                m = re.search(r'[-+]?\d+\.\d+', line)
                if m: sharpe = m.group()
            if 'total_return' in line.lower() or '总收益' in line or 'total return' in line.lower():
                m = re.search(r'[-+]?\d+\.\d+', line)
                if m: ret = m.group()

        if proc.returncode == 0:
            print(f"OK ({elapsed:.1f}s) sharpe={sharpe} ret={ret}")
            results.append({'name': name, 'status': 'ok', 'sharpe': sharpe, 'ret': ret, 'time': elapsed})
        else:
            err_short = err.strip().split('\n')[-1] if err.strip() else 'unknown'
            print(f"FAIL ({elapsed:.1f}s): {err_short[:80]}")
            results.append({'name': name, 'status': 'fail', 'error': err_short[:80]})
    except subprocess.TimeoutExpired:
        print("TIMEOUT (>300s)")
        results.append({'name': name, 'status': 'timeout'})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({'name': name, 'status': 'error', 'error': str(e)[:80]})

print(f"\n{'='*80}")
ok = [r for r in results if r['status'] == 'ok']
fail = [r for r in results if r['status'] != 'ok']
print(f"完成: {len(ok)} 成功, {len(fail)} 失败")
for r in fail:
    print(f"  {r['name']}: {r.get('error', r['status'])}")
