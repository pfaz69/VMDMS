# Code and algorithm design by Paolo Fazzini (paolo.fazzini@cnr.it)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import matplotlib.patches as patches
from itertools import combinations
from scipy.optimize import curve_fit



def plot_graphs(num_tests, models, target_graph, ensemble, label_ensemble, file_out, halos, do_show=False):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    line_styles = ['-', '--', '-.', ':']
    font_size = 16
    plt.figure(figsize=(12, 9))
    
    for j in range(num_tests):
        # Create a new scaler for each test
        scaler = MinMaxScaler()
        # Fit the scaler to the target graph

        for i, model in enumerate(models):
            # Plot the forecast for the original signal
            pd = model.predict_direct[j].copy()  # Shape (N, M)
            pr = model.predict_recursive[j].copy()  # Shape (N, M)
            

            # Aggregate plots for each column
            for k in range(pd.shape[1]):
                plt.plot(pd[:, k], color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], label=f"{model.labels[0]} (col {k+1})")
                plt.plot(pr[:, k], color=colors[(i + len(models)) % len(colors)], linestyle=line_styles[(i + len(models)) % len(line_styles)], label=f"{model.labels[1]} (col {k+1})")
            
            if halos[i] is not None:
                for k in range(pr.shape[1]):
                    plt.fill_between(
                        np.arange(len(pr[:, k])),
                        np.squeeze(pr[:, k] - halos[i][:, k]),
                        np.squeeze(pr[:, k] + halos[i][:, k]),
                        color=colors[i % len(colors)],
                        alpha=0.2
                    )

        tg = target_graph[j].copy()  # Shape (N, 1)
        plt.plot(tg, color='k', linestyle='-', label='target')

        if len(models) > 1:
            en = ensemble[j].copy()  # Assuming ensemble has shape (N, 1)
            plt.plot(en, color='c', linestyle='-', label=label_ensemble)

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        # plt.legend()
        plt.savefig(f"{file_out}_test_{j+1}.png", dpi=500)
        if do_show:
            plt.show()
        plt.close()


def plot_losses(models, file_out, do_show = False):
    plt.figure(figsize=(10, 6))

    for model in models:
        if model.loss_history.history['loss'] is not None:
            plt.plot(model.loss_history.history['loss'], label=model.labels[0])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(file_out)
    if do_show:
        plt.show()
    plt.close()


def plot_RMSEs(num_tests, models, target_graph, ensemble, file_out, do_show = False):
    labels = []
    rmse_values = []
    for j in range(num_tests):
    
        for model in models:
            if len(model.predict_direct[j]) > 0:
                labels.append(model.labels[0])
                rmse_values.append(root_mean_squared_error(target_graph[j], model.predict_direct[j][:, 0]))
            
            labels.append(model.labels[1])
            rmse_values.append(root_mean_squared_error(target_graph[j], model.predict_recursive[j][:, 0]))
        
        labels += ['ensemble']
        rmse_values += [root_mean_squared_error(target_graph[j], ensemble[j][:, 0])]
        
        x = np.arange(len(labels))
        width = 0.6

        fig, ax = plt.subplots()
        rects = ax.bar(x, rmse_values, width, color = ['b', 'g']*(len(labels)//2) + ['c'])

        ax.set_ylabel('RMSE')
        ax.set_title('RMSEs of Models')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                    '%.2f' % float(height),
                    ha='center', va='bottom', color='yellow')

        fig.tight_layout()
        fig.set_size_inches(15, 9)
        plt.savefig(f"{file_out}_test_{j+1}", dpi=500)
        if do_show:
            plt.show()
        plt.close()




def plot_N_graphs(i_test, graphs, labels, dir_plot, model_type, type_graph, val_extra_normalization_for_latex = None, do_show = False):

    # This is temporary to highight our results
    line_styles = ['dashed', 'dotted', 'dashdot', ':', 'dashed', 'solid']
    #line_thick = [1, 1, 1, 1, 3, 3]
    line_thick = [1, 1, 3, 3]
    font_size = 20
    scaler = MinMaxScaler()

    plt.figure(figsize=(12,6))


    for i in range(len(graphs)):
        data = graphs[i].copy()
        if val_extra_normalization_for_latex is not None:
            data = data.squeeze()/val_extra_normalization_for_latex
        plt.plot(data, label=labels[i], linestyle=line_styles[i%len(graphs)], linewidth=line_thick[i%len(graphs)])


    plt.legend(prop={'size': font_size - 3})
    plt.xlabel("Time Steps", fontsize=font_size)
    plt.ylabel("Power", fontsize=font_size) 
    plt.xticks(fontsize=font_size - 3)
    plt.yticks(fontsize=font_size - 3)
    if dir_plot is not None:
        plt.savefig(f"{dir_plot}/{model_type}_vmd_{type_graph}_{i_test}", dpi=500)

    if do_show:
        plt.show()
    plt.close()




def plot_cumulative_RMSEs(signals, target, labels, file, threshold = None, cap_func = None, do_graph=False, do_show=False):
    """
    Compute RMSE, MAE, SMAPE, MASE, and RMSSE for multiple signals, and optionally plot RMSE.
    """

    if threshold is None:
        # Calculate RMSE for each signal
        rmse_values = [np.sqrt(np.mean((signal - target) ** 2)) for signal in signals]

        # Calculate MAE for each signal
        mae_values = [np.mean(np.abs(signal - target)) for signal in signals]

        # Calculate SMAPE for each signal
        smape_values = [100 * np.mean(np.abs(signal - target) / ((np.abs(signal) + np.abs(target)) / 2)) for signal in signals]

        # Calculate denominator for MASE and RMSSE (naive forecast based on persistence/random walk)
    else:
        # Compute RMSE while ignoring elements where target[i, 0] < threshold
        rmse_values = [
            np.sqrt(np.mean((signal[target[:, 0] >= threshold] - target[target[:, 0] >= threshold]) ** 2))
            for signal in signals
        ]

        # Compute MAE while ignoring elements where target[i, 0] < threshold
        mae_values = [
            np.mean(np.abs(signal[target[:, 0] >= threshold] - target[target[:, 0] >= threshold]))
            for signal in signals
        ]

        # Compute SMAPE while ignoring elements where target[i, 0] < threshold
        smape_values = [
            100 * np.mean(
                np.abs(signal[target[:, 0] >= threshold] - target[target[:, 0] >= threshold])
                / ((np.abs(signal[target[:, 0] >= threshold]) + np.abs(target[target[:, 0] >= threshold])) / 2)
            )
            for signal in signals
        ]



    naive_forecast = np.roll(target, 1)  # Shift target signal by 1 (random walk)
    
    naive_error = np.abs(target[1:] - naive_forecast[1:]) 
    mase_denominator = np.mean(naive_error)  
    rmsse_denominator = np.sqrt(np.mean(naive_error ** 2))  

    # Calculate MASE and RMSSE for each signal
    mase_values = [np.mean(np.abs(signal[1:] - target[1:]) / mase_denominator) for signal in signals]
    rmsse_values = [np.sqrt(np.mean((signal[1:] - target[1:]) ** 2)) / rmsse_denominator for signal in signals]

    if do_graph:
        # Create a bar plot for RMSE
        plt.figure(figsize=(10, 6))
        if labels is not None:
            plt.bar(labels, rmse_values, color='blue', alpha=0.7)
        plt.xticks(fontsize=16)
        plt.grid(True)

        # Save the plot to the specified file
        if file is not None:
            plt.savefig(file)
        if do_show:
            plt.show()
        plt.close()

    return rmse_values, mae_values, smape_values, mase_values, rmsse_values




def compute_n_plot_spectrum(stream_hatmodes, stream_freqs, file_spectrum, do_show=False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF6347', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0']
    font_size = 20
    num_plots = len(stream_hatmodes)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 8*num_plots))
    bandwidth_lst = []

    for i in range(num_plots):
        u_hat = stream_hatmodes[i]
        omega = stream_freqs[i]
        K = u_hat.shape[0]
        T = u_hat.shape[1]
        t = np.arange(1, T+1) / T
        freqs = t - 0.5 - (1/T)
        if num_plots > 1:
            axis = axs[i]
        else:
            axis = axs
        for k in range(K):
            # plot the spectrum
            axis.plot(freqs, np.abs(u_hat[k,:]), label=f'Mode {k+1}', color=colors[k % len(colors)], alpha=0.3)

            # plot the center frequency
            axis.axvline(x=omega[k], color=colors[k % len(colors)], linestyle='--')

            # compute the power spectrum
            power_spectrum = np.abs(u_hat[k,:])**2

            # compute the weighted mean frequency
            mean_freq = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

            # compute the standard deviation of the frequency distribution
            bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * power_spectrum) / np.sum(power_spectrum))
            bandwidth_lst.append(bandwidth)

            # plot the bandwidth as a translucent column centered at omega
            axis.bar(omega[k], height=np.max(power_spectrum), width=(2/K)*bandwidth, color=colors[k % len(colors)], alpha=0.1)

        if i == 0:
            axis.set_ylabel('Amplitude', fontsize=font_size)
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:0.0e}'.format(x)))  # Format y-axis in scientific notation
            axis.set_xlim([0, 0.5])
            axis.set_ylim([0, 20000])
            axis.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
            axis.tick_params(axis='y', labelsize=font_size-2)
        elif i == num_plots-1:
            axis.set_ylabel('')
            axis.set_xlim([0, 0.5])
            axis.set_ylim([0, 20000])
            axis.tick_params(axis='y', labelleft=False)  # Hide y-axis labels
            axis.tick_params(axis='x', labelsize=font_size-2)
        else:
            axis.set_ylabel('')
            axis.set_xlim([0, 0.5])
            axis.set_ylim([0, 20000])
            axis.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
            axis.tick_params(axis='y', labelleft=False)  # Hide y-axis labels

    fig.text(0.5, 0.985, 'VMD Modes (Spectrum and Bandwidth)', ha='center', fontsize=font_size)  # Adjust y position for individual titles
    fig.text(0.06, 0.008, 'Frequency', ha='center', fontsize=font_size)  # Adjust y position for individual titles
    plt.tight_layout(pad=5)  # Increase margin between subplots
    plt.savefig(file_spectrum)
    if do_show:
        plt.show()
    plt.close()
    return bandwidth_lst




def plot_dots_and_subsets(data, subsets, eps, file_subsets=None, do_show = False):
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF6347', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    fig, ax = plt.subplots()
    ax.yaxis.set_visible(False)  # Remove y-axis ticks
    
    # Plot each subset in a different color
    for i, vec in enumerate(list(zip(*data))):
        plt.scatter(vec, [0]*len(vec), color=colors[i % len(colors)], s=5)

    for i, subset in enumerate(subsets):
        # Create an ellipse
        center = (np.mean(subset), 0)
        width = max(subset) - min(subset) + eps
        height = width
        ellipse = patches.Ellipse(center, width, height, edgecolor='none', facecolor='black', alpha=0.2)  # Fill the ellipse with alpha=0.2
        
        # Add the ellipse to the plot
        ax.add_patch(ellipse)
        plt.text(center[0], center[1]+.01, str(len(subset)), horizontalalignment='center', verticalalignment='center', fontsize=10, color='red')
    
    plt.axis('equal')  # Add this line to make x and y axis equal
    
    if file_subsets != None:
        plt.savefig(file_subsets)
    if do_show:
        plt.show()
    plt.close()


 

def plot_main_graphs(
                            k_mode,
                            num_tests,
                            models,
                            target_graph,
                            ensemble,
                            label_ensemble,
                            dir_base_1,
                            dir_base_2,
                            dir_base_3
    ):

    # Plot the graphs
    halos = [None]*len(models)
    plot_graphs(
        num_tests,
        models=models,
        target_graph=target_graph,
        ensemble=ensemble,
        label_ensemble=label_ensemble,
        file_out=f"{dir_base_1}/mode_{k_mode}",
        halos=halos
    )

    # Plot RMSEs
    plot_RMSEs(
        num_tests,
        models=models,
        target_graph=target_graph,
        ensemble=ensemble,
        file_out=f"{dir_base_2}/mode_{k_mode}"
    )

    # Plot losses
    plot_losses(
        models=models,
        file_out=f"{dir_base_3}/mode_{k_mode}"
    )












