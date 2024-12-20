import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'task': ['Task A', 'Task B', 'Task C'],
    'start': [0, 2, 5],
    'end': [3, 7, 8]
})

# Create figure and axes
fig, ax = plt.subplots()

# Plot Gantt chart
ax.broken_barh([(start, end - start) for start, end in zip(df['start'], df['end'])],
                (0, 1), facecolors='blue')
ax.set_yticks([0.5])
ax.set_yticklabels(df['task'])
ax.set_xlabel('Time')
ax.set_title('Gantt Chart')
plt.show()