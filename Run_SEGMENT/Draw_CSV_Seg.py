# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
segment_data = pd.read_csv("results.csv")
# Strip leading/trailing whitespace from column names
segment_data.columns = segment_data.columns.str.strip()

# Display the first few rows of the segmentation data
segment_data.head()

# Changing the default font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Creating a figure for training loss with adjusted size
fig, ax = plt.subplots(figsize=(10, 8))
# Plotting training loss metrics for segmentation
ax.plot(segment_data['epoch'], segment_data['train/box_loss'], label='Box Loss', color='purple')
ax.plot(segment_data['epoch'], segment_data['train/seg_loss'], label='Seg Loss', color='red')
ax.plot(segment_data['epoch'], segment_data['train/cls_loss'], label='Cls Loss', color='brown')
ax.plot(segment_data['epoch'], segment_data['train/dfl_loss'], label='Dfl Loss', color='pink')
ax.set_xlabel('Epoch', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title('Training Loss over Epochs (Segmentation)', fontsize=20)
ax.legend(fontsize=17)
ax.grid(True)
ax.set_ylim(0, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()

# Creating a figure for validation loss with adjusted size
fig, ax = plt.subplots(figsize=(10, 8))
# Plotting validation loss metrics for segmentation
ax.plot(segment_data['epoch'], segment_data['val/box_loss'], label='Box Loss')
ax.plot(segment_data['epoch'], segment_data['val/seg_loss'], label='Seg Loss')
ax.plot(segment_data['epoch'], segment_data['val/cls_loss'], label='Cls Loss')
ax.plot(segment_data['epoch'], segment_data['val/dfl_loss'], label='Dfl Loss')
ax.set_xlabel('Epoch', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title('Validation Loss over Epochs (Segmentation)', fontsize=20)
ax.legend(fontsize=17)
ax.grid(True)
ax.set_ylim(0, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()

# Create a figure for metrics
fig, ax = plt.subplots(figsize=(10, 8))
# Plotting metrics for segmentation (using B metrics)
ax.plot(segment_data['epoch'], segment_data['metrics/precision(B)'], label='Precision')
ax.plot(segment_data['epoch'], segment_data['metrics/recall(B)'], label='Recall')
ax.plot(segment_data['epoch'], segment_data['metrics/mAP50(B)'], label='mAP50')
ax.plot(segment_data['epoch'], segment_data['metrics/mAP50-95(B)'], label='mAP50-95')
ax.set_xlabel('Epoch', fontsize=18)
ax.set_ylabel('Metric Score', fontsize=18)
ax.set_title('Metrics over Epochs (Segmentation)', fontsize=20)
ax.legend(fontsize=17)
ax.grid(True)
ax.set_ylim(0, 1.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()
