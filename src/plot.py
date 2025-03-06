from recurrent.mytypes import *
from recurrent.parameters import *
from matplotlib import pyplot as plt
import jax.numpy as jnp

like = AllLogs(
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000, 1)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000, 1)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000,)),
    jax.numpy.empty((1000, 1186), dtype=jnp.complex64),
)

logs = eqx.tree_deserialise_leaves("src/mytest55.eqx", like)

# Create the figure and subplots
fig, (ax1, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(
    8, 1, figsize=(12, 20), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]}
)

# First subplot (Losses and Learning Rate)
ax1.plot(logs.trainLoss, label="Train Loss", color="blue")
ax1.plot(logs.validationLoss, label="Validation Loss", color="red")
ax1.plot(logs.testLoss, label="Test Loss", color="green")

ax2 = ax1.twinx()
ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")

# Labels and legends for the first subplot
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Learning Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Training Progress")

# Second subplot (Parameter Norm)
ax3.plot(logs.parameterNorm, label="Parameter Norm", color="orange")
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Parameter Norm")
ax3.legend(loc="upper right")
ax3.set_title("Parameter Norm Over Time")

# Third subplot (Oho Gradient)
ax4.plot(logs.ohoGradient, label="Oho Gradient", color="cyan")
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Oho Gradient")
ax4.legend(loc="upper right")
ax4.set_title("Oho Gradient Over Time")

# Fourth subplot (Train Gradient)
ax5.plot(logs.trainGradient, label="Train Gradient", color="magenta")
ax5.set_xlabel("Epochs")
ax5.set_ylabel("Train Gradient")
ax5.legend(loc="upper right")
ax5.set_title("Train Gradient Over Time")

# Fifth subplot (Validation Gradient)
ax6.plot(logs.validationGradient, label="Validation Gradient", color="brown")
ax6.set_xlabel("Epochs")
ax6.set_ylabel("Validation Gradient")
ax6.legend(loc="upper right")
ax6.set_title("Validation Gradient Over Time")

# Sixth subplot (Immediate Influence Tensor)
ax7.plot(logs.immediateInfluenceTensor, label="Immediate Influence Tensor", color="teal")
ax7.set_xlabel("Epochs")
ax7.set_ylabel("Immediate Influence Tensor")
ax7.legend(loc="upper right")
ax7.set_title("Immediate Influence Tensor Over Time")

# Seventh subplot (Influence Tensor)
ax8.plot(logs.influenceTensor, label="Influence Tensor", color="darkblue")
ax8.set_xlabel("Epochs")
ax8.set_ylabel("Influence Tensor")
ax8.legend(loc="upper right")
ax8.set_title("Influence Tensor Over Time")

# Eighth subplot (Hessian)
ax9.plot(logs.hessian, label="Hessian", color="limegreen")
ax9.set_xlabel("Epochs")
ax9.set_ylabel("Hessian")
ax9.legend(loc="upper right")
ax9.set_title("Hessian Over Time")

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep the main title from overlapping
plt.savefig("src/mytest55_clean.png", dpi=300)
plt.close()
