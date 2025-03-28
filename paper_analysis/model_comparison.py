import matplotlib.pyplot as plt

# Data
categories = ['Corn leaf Spot', 'Corn leaves', 'Background']
ours_scores = [0.852, 0.960, 0.967]
original_model_scores = [0.838, 0.949, 0.960]

# Plotting
plt.plot(categories, ours_scores, marker='o', label='Ours')
plt.plot(categories, original_model_scores, marker='o', label='Original model')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('IOU')
plt.legend()

# Annotate data points
for i in range(len(categories)):
    plt.text(categories[i], ours_scores[i], f"{ours_scores[i]:.3f}", ha='center', va='bottom')
    plt.text(categories[i], original_model_scores[i], f"{original_model_scores[i]:.3f}", ha='center', va='bottom')

# Show plot
plt.grid(True)
plt.savefig('model_comparison1.png')
plt.show()
