from recurrent.mytypes import *
from recurrent.parameters import *
from matplotlib import pyplot as plt

like = AllLogs(jax.numpy.empty((1000,)), jax.numpy.empty((1000,)), jax.numpy.empty((1000,)), jax.numpy.empty((1000, 1)))

logs = eqx.tree_deserialise_leaves("src/rtrl_100000_100_t1_15_t2_17_test1_mlr_e3_lr_4e1.eqx", like)
# print(logs.trainLoss[0])
# quit()

plt.figure(figsize=(12, 6))
plt.plot(logs.trainLoss, label="Train Loss", color="blue")
plt.plot(logs.validationLoss, label="Validation Loss", color="red")
plt.plot(logs.testLoss, label="Test Loss", color="green")

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Learning Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.set_ylim(0.5, 1.2)  # Adjust as needed
ax2.set_ylim(0, 0.8)  # Adjust based on learning rate scale


plt.title("Training Progress")
plt.savefig("src/rtrl_100000_100_t1_15_t2_17_test1_mlr_e3_lr_4e1_clean.png", dpi=300)
plt.close()

print(logs.learningRate[:100])
