#!/usr/bin/env python3
"""
自动处理新的TQ期权数据 + 重启API服务器
用法: python3 scripts/auto_process_options.py
可以加到 crontab 每天盘后运行
"""
import os, sys, subprocess, time

BASE_DIR = os.path.expanduser("~/home/futures_platform")
SCRIPT = os.path.join(BASE_DIR, "scripts/process_tq_options.py")

def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 开始处理期权数据...")
    result = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, cwd=BASE_DIR)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print(f"处理失败 (exit {result.returncode})")
        return

    # 重启API服务器
    print("重启API服务器...")
    api_script = os.path.join(BASE_DIR, "dashboard/api_server.py")
    # Kill existing
    os.system("kill $(lsof -ti:5001) 2>/dev/null")
    time.sleep(1)
    # Start new
    subprocess.Popen([sys.executable, api_script], cwd=os.path.join(BASE_DIR, "dashboard"))
    print("完成!")

if __name__ == '__main__':
    main()
