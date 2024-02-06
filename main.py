import subprocess
import re
import matplotlib.pyplot as plt

# Function to call the Rust program using 'cargo run --release' and parse its output
def call_and_parse_cargo():
    try:
        # Call the Rust program with 'cargo run --release'
        result = subprocess.run(['cargo', 'run', '--release'], capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse the output using regular expressions
        heuristic_match = re.search(r"Heuristic: (\d+) steps in ([\d.]+)ms", output)
        value_iteration_match = re.search(r"Value iteration: (\d+) steps in ([\d.]+)ms", output)

        if heuristic_match:
            heuristic_steps = int(heuristic_match.group(1))
            heuristic_time = float(heuristic_match.group(2))
        else:
            heuristic_steps = heuristic_time = None

        if value_iteration_match:
            value_iteration_steps = int(value_iteration_match.group(1))
            value_iteration_time = float(value_iteration_match.group(2))
        else:
            value_iteration_steps = value_iteration_time = None

        return heuristic_steps, heuristic_time, value_iteration_steps, value_iteration_time
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

# Lists to store data
heuristic_steps_list = []
heuristic_time_list = []
value_iteration_steps_list = []
value_iteration_time_list = []

# Call 'cargo run --release' 100 times and store the results
for _ in range(100):
    print(_)
    h_steps, h_time, v_steps, v_time = call_and_parse_cargo()
    if h_steps is not None and h_time is not None:
        heuristic_steps_list.append(h_steps)
        heuristic_time_list.append(h_time)
    if v_steps is not None and v_time is not None:
        value_iteration_steps_list.append(v_steps)
        value_iteration_time_list.append(v_time)

# Check if we have data to plot
if heuristic_steps_list and heuristic_time_list and value_iteration_steps_list and value_iteration_time_list:
    # Plotting the results
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
