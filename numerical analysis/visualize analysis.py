import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def forward_difference(data):
    differences = np.diff(data, n=1)
    return differences

def backward_difference(data):
    differences = np.diff(data[::-1], n=1)[::-1]
    return differences

def central_difference(data):
    forward = forward_difference(data)
    backward = backward_difference(data)
    central = (forward + backward) / 2
    return central

def visualize_differences(data, forward_diff, backward_diff, central_diff):
    # Create a DataFrame to show the data and differences
    df = pd.DataFrame({
        'Data': data,
        'Forward Difference': np.concatenate(([np.nan], forward_diff)),
        'Backward Difference': np.concatenate(([np.nan], backward_diff)),
        'Central Difference': np.concatenate(([np.nan], central_diff))
    })

    # Display the table
    print("\nDifferences Table:")
    print(df)
    
    # Plotting the differences
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Data", marker='o', linestyle='-', color='blue')
    plt.plot(np.arange(1, len(data)), forward_diff, label="Forward Difference", marker='x', linestyle='--', color='green')
    plt.plot(np.arange(1, len(data)), backward_diff, label="Backward Difference", marker='x', linestyle='--', color='red')
    plt.plot(np.arange(1, len(data)), central_diff, label="Central Difference", marker='x', linestyle='-.', color='purple')

    plt.title("Forward, Backward, and Central Differences")
    plt.xlabel("Index")
    plt.ylabel("Difference Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Input data
    data = list(map(float, input("Enter data points separated by spaces: ").split()))

    # Calculate the differences
    forward_diff = forward_difference(data)
    backward_diff = backward_difference(data)
    central_diff = central_difference(data)

    # Visualize the differences
    visualize_differences(data, forward_diff, backward_diff, central_diff)

if __name__ == "__main__":
    main()
