import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_heatmap(data_dict, vmax, title, outpath):
    # Using the dark theme from seaborn
    sns.set_theme(style="darkgrid", palette="deep")

    # Convert the dictionary to a dataframe
    df = pd.DataFrame({
        'x': [key[0] for key in data_dict.keys()],
        'y': [key[1] for key in data_dict.keys()],
        'value': list(data_dict.values())
    })
    # # Pivot the dataframe to get a matrix-like format
    df_pivot = df.pivot(index='y', columns='x', values='value')
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    color_map = sns.color_palette("flare", as_cmap=True)
    color_map.set_bad('white')
    ax = sns.heatmap(df_pivot, cmap=color_map, annot=False, linewidths=.5, vmax=vmax, vmin=0)
    ax.set(xlabel='', ylabel='', xticks=[], yticks=[])
    # Adding title
    # plt.title(title)
    
    plt.tight_layout()
    plt.savefig(outpath)

def plot_smoothed_line(data_dict, xlabel, ylabel, title, outpath):
    # Using the dark theme from seaborn
    sns.set_theme(style="darkgrid", palette="deep")

    # Extracting values from the dictionary
    counts = np.array(list(data_dict.values()))

    # Creating the figure
    plt.figure(figsize=(10, 6))

    # Line plot
    sns.lineplot(x=range(len(counts)), y=counts, label="Counts", alpha=0.8)
    
    # Overlaying a KDE smoothed curve
    # sns.kdeplot(counts, cumulative=True, bw_adjust=0.5, label="Smoothed", color="red")

    # Adding title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # plt.legend()

    plt.tight_layout()
    plt.savefig(outpath)

def plot_kde(data_dict, xlabel, ylabel, title, outpath):
    # Using the dark theme from seaborn
    sns.set_theme(style="darkgrid", palette="deep")

    # Extracting values from the dictionary
    counts = np.array(list(data_dict.values()))

    # Creating the KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(counts, fill=True)

    # Adding title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(outpath)

def plot_dict_values_hist(data_dict, xlabel, ylabel, title, outpath):
    # Using the dark theme from seaborn
    sns.set_theme(style="darkgrid", palette="deep")

    # Extracting keys and values from the dictionary
    categories = [' & '.join(pair) for pair in data_dict.keys()]
    counts = list(data_dict.values())
    
    # Creating the histogram
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=counts)
 
    # Adding title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig(outpath)
    
def plot_proportion_bar_chart(proportions, xlabel, ylabel, title, outpath):
    # Set the dark theme
    sns.set_theme(style="darkgrid", palette="deep")

    # Data setup for stacked bar
    categories = list(proportions.keys())
    values = list(proportions.values())

    # Create a single bar chart with proportions
    plt.figure(figsize=(6, 6))
    bottom = 0
    colors = sns.color_palette("tab10", len(proportions))

    for idx, category in enumerate(categories):
        plt.bar(1, values[idx], bottom=bottom, label=category, color=colors[idx])
        bottom += values[idx]

    # Customize the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xticks([])  # Remove x-axis tick as there's only one bar
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Show the plot
    plt.tight_layout()
    plt.savefig(outpath)