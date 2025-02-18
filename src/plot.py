from recurrent.mytypes import *
from recurrent.parameters import *
from matplotlib import pyplot as plt

like = AllLogs(
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000, 1)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000, 1)),
)

logs = eqx.tree_deserialise_leaves("src/mytest39.eqx", like)
# print(logs.trainLoss[0])
# quit()

fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})

# First subplot (Losses and Learning Rate)
ax1.plot(logs.trainLoss, label="Train Loss", color="blue")
ax1.plot(logs.validationLoss, label="Validation Loss", color="red")
ax1.plot(logs.testLoss, label="Test Loss", color="green")

ax2 = ax1.twinx()
ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")

# Labels and legends
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Learning Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Training Progress")

# Set axis limits (optional)
ax1.set_ylim(0.4, 0.8)
ax2.set_ylim(0, 0.4)

# Second subplot (Parameter Norm)
ax3.plot(logs.parameterNorm, label="Parameter Norm", color="orange")
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Parameter Norm")
ax3.legend(loc="upper right")
ax3.set_title("Parameter Norm Over Time")
ax3.set_ylim(0, 10)

# Third subplot (ohoGradient)
ax4.plot(logs.ohoGradient, label="Oho Gradient", color="cyan")
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Oho Gradient")
ax4.legend(loc="upper right")
ax4.set_title("Oho Gradient Over Time")
ax4.set_ylim(-20, 20)
# ax4.set_ylim(0, max(logs.ohoGradient) * 1.1)  # Optional scaling

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep the main title from overlapping
plt.savefig("src/mytest39_clean.png", dpi=300)
plt.close()

print(logs.learningRate[:300])
