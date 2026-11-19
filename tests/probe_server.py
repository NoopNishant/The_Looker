"""
tests/test_server_probe.py - Spin up server on localhost:8000, probe root & endpoints, and exit.
"""

import time
import subprocess
import urllib.request
import json
import sys

def probe_server():
    print("[Probe] Starting run_web.py...")
    proc = subprocess.Popen([sys.executable, "run_web.py"])
    time.sleep(5) # Allow uvicorn to bind

    try:
        # Probe GET /
        url = "http://127.0.0.1:8000/"
        print(f"[Probe] Fetching {url}...")
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.getcode()
            html = response.read().decode("utf-8")
            print(f"[Probe] Root endpoint status: {status} (HTML bytes: {len(html)})")
            assert status == 200
            assert "The Looker" in html
            assert "Currency Forensics" in html
            print(" [PASS] Root HTML correctly loaded")

        # Probe POST /api/generate-demo
        url_demo = "http://127.0.0.1:8000/api/generate-demo"
        print(f"[Probe] Testing POST {url_demo}...")
        req = urllib.request.Request(
            url_demo,
            data=json.dumps({"denomination": "500", "is_counterfeit": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = json.loads(response.read().decode("utf-8"))
            print(f"[Probe] Demo endpoint status: {status} | Verdict: {data.get('verdict')} | Score: {data.get('composite_counterfeit_score')}")
            assert status == 200
            assert data["verdict"] == "GENUINE"
            print(" [PASS] Synthetic demo API correctly responded")

        print("\nAll server probe checks PASSED successfully!")
    finally:
        print("[Probe] Terminating test server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    probe_server()

