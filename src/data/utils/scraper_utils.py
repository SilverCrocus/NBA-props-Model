#!/usr/bin/env python3
"""
Kill all Chrome processes to fix stuck profile
"""

import subprocess
import time

import psutil


def kill_all_chrome():
    """Kill all Chrome and chromedriver processes"""
    print("🔨 Killing all Chrome processes...")

    killed_count = 0

    # First try with psutil
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            proc_name = proc.info["name"].lower()
            if (
                "chrome" in proc_name or "chromium" in proc_name or "chromedriver" in proc_name
            ):  # noqa: E501
                try:
                    print(
                        f"  Killing: {
                            proc.info['name']} (PID: {
                            proc.info['pid']})"
                    )
                    proc.terminate()
                    killed_count += 1
                except BaseException:
                    pass

        if killed_count > 0:
            time.sleep(2)
            # Force kill any remaining
            for proc in psutil.process_iter(["pid", "name"]):
                proc_name = proc.info["name"].lower()
                if (
                    "chrome" in proc_name or "chromium" in proc_name or "chromedriver" in proc_name
                ):  # noqa: E501
                    try:
                        proc.kill()
                    except BaseException:
                        pass
    except Exception as e:
        print(f"  psutil error: {e}")

    # Also try with subprocess as backup
    try:
        subprocess.run(["pkill", "-", "Chrome"], capture_output=True)
        subprocess.run(["pkill", "-", "chrome"], capture_output=True)
        subprocess.run(["pkill", "-", "chromedriver"], capture_output=True)
    except BaseException:
        pass

    if killed_count > 0:
        print(f"✅ Killed {killed_count} Chrome processes")
        print("   Chrome profile should be unlocked now")
    else:
        print("  No Chrome processes found")

    print("\n✅ You can now run: python3 ctg_robust_scraper.py")


if __name__ == "__main__":
    kill_all_chrome()
