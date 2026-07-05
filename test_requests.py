"""
Two test inputs for the demo video, as required by the assignment:
  1. A standard, well-specified business request.
  2. A complex/ambiguous request with missing info and conflicting
     constraints, forcing the agent to make and record assumptions.

Run the API first:  uvicorn app.main:app --reload
Then:                python test_requests.py
"""
import json
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

STANDARD_REQUEST = (
    "Create meeting minutes for our weekly engineering sync held on July 3rd, 2026. "
    "Attendees: Priya (Eng Lead), Sam (Backend), Jordan (Frontend), Ana (QA). "
    "We discussed the API rate-limiting rollout, a bug in checkout flow, and Q3 hiring plan. "
    "Action items should be assigned to owners with due dates."
)

COMPLEX_REQUEST = (
    "We need something to send the client about our new product before Friday, but I'm not sure if "
    "it should be a proposal or a spec -- honestly whatever gets us the deal fastest. Budget is 'flexible' "
    "but also we were told last month not to go over 50k. The client wants it 'enterprise-grade' but also "
    "wants a two-week turnaround. Just make it look good and cover whatever a serious buyer would expect."
)


def run(label: str, request_text: str):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"REQUEST: {request_text}\n")
    resp = requests.post(f"{BASE_URL}/agent", json={"request": request_text}, timeout=180)
    print(f"HTTP {resp.status_code}")
    data = resp.json()
    print(json.dumps(data, indent=2)[:3000])
    if resp.status_code == 200:
        print(f"\n-> Download the generated document at: {BASE_URL}{data['download_url']}")


if __name__ == "__main__":
    run("TEST 1: STANDARD BUSINESS REQUEST", STANDARD_REQUEST)
    print("\nWaiting 30s before the next test so we don't collide with the "
          "free-tier tokens-per-minute limit on the same request budget...")
    time.sleep(30)
    run("TEST 2: COMPLEX / AMBIGUOUS REQUEST", COMPLEX_REQUEST)
