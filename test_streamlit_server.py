"""
Stage 7 Verification Script: Streamlit Local Server Probe
=========================================================
Starts Streamlit as a local background process on port 8501,
probes the health and main web endpoints, verifies HTTP 200 OK,
and shuts down the server cleanly.
"""

import subprocess
import time
import urllib.request
import sys

def main():
    print("--- Testing Stage 7: Local Streamlit App Startup ---")
    cmd = [
        r".venv\Scripts\streamlit.exe",
        "run",
        "app.py",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.serverAddress=localhost",
    ]

    print("Launching Streamlit app process...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Give Streamlit server 6 seconds to bind port and initialize
        print("Waiting for server to initialize...")
        time.sleep(6)

        # 1. Probe health endpoint
        health_url = "http://localhost:8501/_stcore/health"
        print(f"Probing Health Check endpoint: {health_url} ...")
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            status_code = resp.getcode()
            body = resp.read().decode("utf-8")
            print(f"  Response Status: {status_code} ({body.strip()})")
            assert status_code == 200, f"Expected 200, got {status_code}"

        # 2. Probe main dashboard UI page
        main_url = "http://localhost:8501/"
        print(f"Probing Main Dashboard UI: {main_url} ...")
        with urllib.request.urlopen(main_url, timeout=5) as resp:
            status_code = resp.getcode()
            html = resp.read().decode("utf-8")
            print(f"  Response Status: {status_code} (HTML length: {len(html)} bytes)")
            assert status_code == 200, f"Expected 200, got {status_code}"
            assert "StockScope" in html or "streamlit" in html.lower(), "Streamlit app HTML not found"

        print("\nSTAGE 7 SUCCESS: Streamlit app runs locally, binds to port 8501, and serves HTTP 200 successfully!")

    except Exception as e:
        print(f"\nSERVER TEST FAILED: {e}")
        # Capture stderr if available
        stderr_output = proc.stderr.read() if proc.stderr else ""
        if stderr_output:
            print("Streamlit Stderr Log:\n", stderr_output)
        sys.exit(1)

    finally:
        print("Shutting down test Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Test server stopped cleanly.")

if __name__ == "__main__":
    main()
