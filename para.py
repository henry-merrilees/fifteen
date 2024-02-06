import subprocess
import re
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Function to call the Rust program using 'cargo run --release' and parse its output
def call_and_parse_cargo():
    try:
        print("Running Rust program...")
        # Call the Rust program with 'cargo run --release'
        result = subprocess.run(['cargo', 'run', '--release'], capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse the output using regular expressions
        heuristic_match = re.search(r"Heuristic: (\d+) steps in ([\d.]+)ms", output)
        value_iteration_match = re.search(r"Value iteration: (\d+) steps in ([\d.]+)ms", output)

        if heuristic_match and value_iteration_match:
            return {
                "heuristic_steps": int(heuristic_match.group(1)),
                "heuristic_time": float(heuristic_match.group(2)),
                "value_iteration_steps": int(value_iteration_match.group(1)),
                "value_iteration_time": float(value_iteration_match.group(2))
            }
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    return None

# Lists to store data
results = []

# Use ThreadPoolExecutor to run processes in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(call_and_parse_cargo) for _ in range(100)]
    for future in futures:
        result = future.result()
        if result:
            results.append(result)

# Extract data from results
heuristic_steps_list = [result["heuristic_steps"] for result in results if result]
heuristic_time_list = [result["heuristic_time"] for result in results if result]
value_iteration_steps_list = [result["value_iteration_steps"] for result in results if result]
value_iteration_time_list = [result["value_iteration_time"] for result in results if result]

# Plotting the results
if results:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(heuristic_steps_list, heuristic_time_list, color='blue')
    plt.title("Heuristic: Steps vs Time")
    plt.xlabel("Steps")
    plt.ylabel("Time (ms)")

    plt.subplot(1, 2, 2)
    plt.scatter(value_iteration_steps_list, value_iteration_time_list, color='red')
    plt.title("Value Iteration: Steps vs Time")
    plt.xlabel("Steps")
    plt.ylabel("Time (ms)")

    plt.tight_layout()
    plt.show()
else:
    print("No data was collected.")

