import numpy as np

# Example data
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array(['a', 'b', 'c', 'd', 'e'])
array3 = np.array([10, 20, 30, 40, 50])

# Get the length of the arrays
length = len(array1)

# Create shuffled indices
shuffled_indices = np.random.permutation(length)

# Use shuffled indices to rearrange the values in all three arrays
shuffled_array1 = array1[shuffled_indices]
shuffled_array2 = array2[shuffled_indices]
shuffled_array3 = array3[shuffled_indices]

# Display the results
print("Original Arrays:")
print("Array 1:", array1)
print("Array 2:", array2)
print("Array 3:", array3)

print("\nShuffled Arrays:")
print("Shuffled Array 1:", shuffled_array1)
print("Shuffled Array 2:", shuffled_array2)
print("Shuffled Array 3:", shuffled_array3)
