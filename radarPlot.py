# plotting rules' radar of quality indicators for 4 sets of rules

import pandas as pd
import matplotlib.pyplot as plt

from math import pi


def plotRadar(data):  # based on https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
    df = pd.DataFrame(data)
    df = df.set_index('group')
    df = df.transpose()
    print("\nDf = " + str(df) + "\n")

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)
    N = len(categories)
    print(str(categories))

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    if df.iloc[0] is not None:
        values = df.iloc[0].values.flatten().tolist()
        values += values[:1]
        print("val="+str(df.iloc[0]))
        ax.plot(angles, values, linewidth=1,
                linestyle='solid', label="R" + df.index[0])
        ax.fill(angles, values, 'b', alpha=0.1)

    if df.iloc[1] is not None:
        values = df.iloc[1].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1,
                linestyle='solid', label="R" + df.index[1])
        ax.fill(angles, values, 'r', alpha=0.1)

    if df.iloc[2] is not None:
        values = df.iloc[2].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1,
                linestyle='solid', label="R" + df.index[2])
        ax.fill(angles, values, 'g', alpha=0.1)

    if df.iloc[3] is not None:
        values = df.iloc[3].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1,
                linestyle='solid', label="R" + df.index[3])
        ax.fill(angles, values, 'm', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()
