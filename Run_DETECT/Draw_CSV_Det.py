# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
detect_data = pd.read_csv("results.csv")
# Strip leading/trailing whitespace from column names
detect_data.columns = detect_data.columns.str.strip()

# Display the first few rows of the detection data
detect_data.head()

# Changing the default font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
# Creating a figure for training loss with adjusted size
fig, ax1 = plt.subplots(figsize=(10, 8))
# Plotting training loss metrics for detection
ax1.plot(detect_data['epoch'], detect_data['train/box_loss'], label='Box Loss', color='blue')
ax1.plot(detect_data['epoch'], detect_data['train/cls_loss'], label='Cls Loss', color='orange')
ax1.plot(detect_data['epoch'], detect_data['train/dfl_loss'], label='Dfl Loss', color='green')
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Loss', fontsize=18)
ax1.set_title('Training Loss over Epochs (Detection)', fontsize=20)
ax1.legend(fontsize=17)
ax1.grid(True)
ax1.set_ylim(0, 5)  # Setting y-axis limit to 3 for ax1 only
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(False)
plt.tight_layout()
plt.show()

# Creating a figure for validation loss with adjusted size
fig, ax1 = plt.subplots(figsize=(10, 8))
# Plotting validation loss metrics for detection
ax1.plot(detect_data['epoch'], detect_data['val/box_loss'], label='Box Loss')
ax1.plot(detect_data['epoch'], detect_data['val/cls_loss'], label='Cls Loss')
ax1.plot(detect_data['epoch'], detect_data['val/dfl_loss'], label='Dfl Loss')
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Loss', fontsize=18)
ax1.set_title('Validation Loss over Epochs (Detection)', fontsize=20)
ax1.legend(fontsize=17)
ax1.grid(True)
ax1.set_ylim(0, 5) # Setting y-axis limit to 3 for ax1 only
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(False)
plt.tight_layout()
plt.show()

# Create a figure for metrics
fig, ax1 = plt.subplots(figsize=(10, 8))
# Plotting metrics for detection
ax1.plot(detect_data['epoch'], detect_data['metrics/precision(B)'], label='Precision')
ax1.plot(detect_data['epoch'], detect_data['metrics/recall(B)'], label='Recall')
ax1.plot(detect_data['epoch'], detect_data['metrics/mAP50(B)'], label='mAP50')
ax1.plot(detect_data['epoch'], detect_data['metrics/mAP50-95(B)'], label='mAP50-95')
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Metric Score', fontsize=18)
ax1.set_title('Metrics over Epochs (Detection)', fontsize=20)
ax1.legend(fontsize=17)
ax1.grid(True)
ax1.set_ylim(0, 1) # Setting y-axis limit to 1 for ax1 only
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(False)
plt.tight_layout()
plt.show()
