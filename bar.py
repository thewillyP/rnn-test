#%%
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Ensure pandas is imported
import numpy as np

# Initialize W&B API
api = wandb.Api()

# Define your project name and the metric you are interested in
metric = "test_loss"  # the metric you want to pull

# Fetch all the runs from the project
runs = api.runs("wlp9800-new-york-university/oho-rnn-generalization-gap7")

# Group runs by the custom tag `is_oho`
runs_oho_1 = []
runs_oho_0 = []

# Iterate through runs to separate them based on `is_oho` tag
for run in runs:
    if run.config.get("is_oho") == 1:
        runs_oho_1.append(run)
    elif run.config.get("is_oho") == 0:
        runs_oho_0.append(run)



# Extract the `test_loss` values for each group
test_loss_oho_1 = [run.summary.get(metric) for run in runs_oho_1 if run.summary.get(metric) != 'NaN']
test_loss_oho_0 = [run.summary.get(metric) for run in runs_oho_0 if run.summary.get(metric) != 'NaN']
test_loss_oho_1 = [x for x in test_loss_oho_1 if x < 100]
test_loss_oho_0 = [x for x in test_loss_oho_0 if x < 100]

print(test_loss_oho_1)
print(test_loss_oho_0)

# Create a bar plot to visualize the results
data = {
    "is_oho": ["1"] * len(test_loss_oho_1) + ["0"] * len(test_loss_oho_0),
    "test_loss": test_loss_oho_1 + test_loss_oho_0,
}

df = pd.DataFrame(data)

#%% 
# Create a bar plot using seaborn
plt.figure(figsize=(10, 8))
sns.boxplot(x="is_oho", y="test_loss", data=df, width=0.3)  # Adjust width here
plt.xticks([0, 1], ["Is OHO", "Fixed Learning Rate"], fontsize=24)  # Update x-ticks with new labels
plt.yticks(fontsize=18)
plt.xlabel("")
plt.ylabel("Test Loss", fontsize=24)

# Adjust the x-axis limits to add more whitespace around the plot
# plt.ylim(0.8, 2.)  # This will add space before and after the boxes

# Adjust spacing to center the plot
plt.subplots_adjust(left=0.2, right=0.8)  # Modify left and right to control the space


plt.show()
# %%
