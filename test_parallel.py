import requests
import concurrent.futures
import sys
from collections import Counter
import time

# Server URL and sweep ID (replace with your actual values)
SERVER_URL = "http://cm001:5000"  # Your server
SWEEP_ID = "9c9c2fe9707c4846a1c37ed15d822744"  # Replace with your actual sweep_id


# Function to request a sweep with retry logic
def get_sweep(sweep_id, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(f"{SERVER_URL}/get_sweep/{sweep_id}", timeout=5)
            if response.status_code == 200:
                config = response.json()["config"]
                ts_str = str(config["ts"]).replace(" ", "")  # e.g., "[12,14]"
                test_seed = (
                    config["test_seed"]
                    if isinstance(config["test_seed"], (int, float))
                    else config["test_seed"][0]
                    if isinstance(config["test_seed"], list)
                    else "unknown"
                )
                return f"{config['inner_learning_rate']}-{config['outer_learning_rate']}-{ts_str}-{config['seed']}-{test_seed}"
            elif response.status_code == 404:
                return "No sweeps left"
            else:
                return f"Error: {response.status_code}"
        except requests.Timeout:
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            return "Request dropped: Timeout"
        except requests.ConnectionError:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return "Request dropped: Connection refused"
        except requests.RequestException as e:
            return f"Request failed: {str(e)}"
    return "Request dropped: Max retries exceeded"


# Main test function
def test_parallel_requests(num_requests=2000):
    print(f"Starting {num_requests} parallel requests to {SERVER_URL}/get_sweep/{SWEEP_ID}")

    # Use ThreadPoolExecutor for parallel requests
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2000) as executor:
        future_to_request = {executor.submit(get_sweep, SWEEP_ID): i for i in range(num_requests)}
        for future in concurrent.futures.as_completed(future_to_request):
            result = future.result()
            results.append(result)

    # Analyze results
    result_counts = Counter(results)
    print("\nResults breakdown (top 10 and errors):")
    for result, count in sorted(result_counts.items())[:10]:
        print(f"{result}: {count} times")
    if "No sweeps left" in result_counts:
        print(f"No sweeps left: {result_counts['No sweeps left']} times")
    for result, count in result_counts.items():
        if result.startswith("Error") or result.startswith("Request dropped") or result.startswith("Request failed"):
            print(f"{result}: {count} times")

    # Verify correctness
    sweep_configs = [r for r in results if "-" in r]  # Filter actual configs
    unique_sweeps = len(set(sweep_configs))
    no_sweeps_left = result_counts.get("No sweeps left", 0)
    errors = sum(
        count
        for result, count in result_counts.items()
        if result.startswith("Error") or result.startswith("Request dropped") or result.startswith("Request failed")
    )
    completed_requests = len(results)

    print(f"\nTotal requests sent: {num_requests}")
    print(f"Total requests completed: {completed_requests}")
    print(f"Total unique sweep configs returned: {unique_sweeps}")
    print(f"'No sweeps left' responses: {no_sweeps_left}")
    print(f"Errors/Dropped: {errors}")

    if completed_requests == num_requests and unique_sweeps == 2000 and no_sweeps_left == 0 and errors == 0:
        print("Test passed: All 2000 sweeps distributed correctly, no duplicates, no drops.")
    else:
        print("Test failed: Check for dropped requests, duplicates, or missing sweeps.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SWEEP_ID = sys.argv[1]
    test_parallel_requests()
