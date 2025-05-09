import numpy as np
import matplotlib.pyplot as plt


def Get_Best_MUE(all_U, all_result, plot=True, figure_path='results/MUE.png'):
    """
    Calculate the Misclassification Uncertainty Error (MUE) and find the optimal threshold.

    Args:
        all_U (np.ndarray or list): Uncertainty values for all samples.
        all_result (np.ndarray or list): Results (1 for correct, 0 for incorrect) for all samples.
        plot (bool, optional): Whether to plot the MUE curve. Defaults to True.
        figure_path (str, optional): Path to save the MUE plot. Defaults to 'results/MUE.png'.

    Returns:
        tuple: (min_mue_value, min_mue_threshold)
            - min_mue_value (float): The minimum MUE value.
            - min_mue_threshold (float): The optimal threshold corresponding to the minimum MUE.
    """
    if type(all_U) is list and type(all_result) is list:
        all_U = np.concatenate(all_U, axis=0)
        all_result = np.concatenate(all_result, axis=0)

    def calculate_mue(threshold, U, result):
        """
        Calculate the MUE for a given threshold.

        Args:
            threshold (float): The threshold to use for calculating MUE.
            U (np.ndarray): Uncertainty values.
            result (np.ndarray): Results (1 for correct, 0 for incorrect).

        Returns:
            float: The MUE value.
        """
        D_c = U[result == 1]  # Uncertainty values for correctly classified samples
        D_i = U[result == 0]  # Uncertainty values for incorrectly classified samples

        if len(D_c) > 0:
            term_c = 0.5 * (np.sum(D_c > threshold) / len(D_c))
        else:
            term_c = 0  # If no correct predictions, set this term to 0

        if len(D_i) > 0:
            term_i = 0.5 * (np.sum(D_i <= threshold) / len(D_i))
        else:
            term_i = 0  # If no incorrect predictions, set this term to 0

        MUE = term_c + term_i
        return MUE  # MUE value

    # Generate a range of thresholds
    thresholds = np.linspace(0, 1, 10000)
    mues = [calculate_mue(t, all_U, all_result) for t in thresholds]

    # Find the threshold corresponding to the minimum MUE
    min_mue_index = np.argmin(mues)
    min_mue_threshold = thresholds[min_mue_index]
    min_mue_value = mues[min_mue_index]

    # print(f"Minimum MUE value: {min_mue_value}, corresponding threshold: {min_mue_threshold}")
    if plot:
        # Plot the MUE curve
        plt.plot(thresholds, mues, color='orange', label='MUE')
        plt.axvline(x=min_mue_threshold, color='red', linestyle='--',
                    label=f'Min MUE Threshold: {min_mue_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('MUE')
        plt.title('MUE vs Threshold')
        plt.legend()
        plt.savefig(figure_path)
        plt.close()
    return min_mue_value, min_mue_threshold