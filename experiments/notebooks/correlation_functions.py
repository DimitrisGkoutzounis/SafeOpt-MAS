import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# Define a function to find continuous positive sections
def find_continuous_positive_sections(x, y,constraint):
    sections = []
    start_idx = None
    for i in range(len(y)):
        if y[i] > constraint and start_idx is None:
            start_idx = i
        elif y[i] <= constraint and start_idx is not None:
            sections.append((x[start_idx:i], y[start_idx:i]))
            start_idx = None
    if start_idx is not None:
        sections.append((x[start_idx:], y[start_idx:]))
    return sections

def find_correlation(sections_y,sections_y2):
    correlations = []
    for x_sec_y, y_sec_y in sections_y:
        x_min, x_max = x_sec_y[0], x_sec_y[-1]
        for x_sec_y2, y_sec_y2 in sections_y2:
            if x_sec_y2[0] <= x_max and x_sec_y2[-1] >= x_min:
                common_x_min = max(x_min, x_sec_y2[0])
                common_x_max = min(x_max, x_sec_y2[-1])
                common_x = np.linspace(common_x_min, common_x_max, num=100)  # 50 points for the correlation
                interp_y = interp1d(x_sec_y, y_sec_y, kind='linear', fill_value="extrapolate")(common_x)
                interp_y2 = interp1d(x_sec_y2, y_sec_y2, kind='linear', fill_value="extrapolate")(common_x)
                corr, _ = pearsonr(interp_y, interp_y2)
                correlations.append(corr)
    return correlations

def generate_correlated_plot(x, y, x2, y2,func_form, sections_y, sections_y2):
    """
    x: x values for the original function
    y: y values for the original function
    x2: x values for the function to be correlated
    y2: y values for the function to be correlated
    func_form: the form of the function to be correlated.Used for the plot legend
    sections_y: the continuous positive sections of the original function
    sections_y2: the continuous positive sections of the function to be correlated
    """
    # Generate the plot for the correlated sections
    plt.figure(figsize=(14, 7))

    # Plot the original functions
    plt.plot(x, y, label='y = sin(x^3) + cos(x^2) - sin(x)')
    plt.plot(x2, y2, label=func_form, linestyle='--')

    corr = find_correlation(sections_y,sections_y2)

    # We interpolate and correlate only if we have matching sections
    for x_sec_y, y_sec_y in sections_y:
        # Find the x range for the current y section
        x_min, x_max = x_sec_y[0], x_sec_y[-1]

        # For each y2 section, check if the x range overlaps with the current y section
        for x_sec_y2, y_sec_y2 in sections_y2:
            if x_sec_y2[0] <= x_max and x_sec_y2[-1] >= x_min:
                # The sections overlap, find the common range
                common_x_min = max(x_min, x_sec_y2[0])
                common_x_max = min(x_max, x_sec_y2[-1])

                # Interpolate both sections to a common x range for comparison
                common_x = np.linspace(common_x_min, common_x_max, num=100)  # 50 points for the correlation
                interp_y = interp1d(x_sec_y, y_sec_y, kind='linear', fill_value="extrapolate")(common_x)
                interp_y2 = interp1d(x_sec_y2, y_sec_y2, kind='linear', fill_value="extrapolate")(common_x)

                # Plot the interpolated, correlated sections
                plt.plot(common_x, interp_y, label='Interpolated y (Correlated Section)', linewidth=2, marker='o')
                plt.plot(common_x, interp_y2, label='Interpolated {func_form} (Correlated Section)'.format(func_form=func_form), linewidth=2, marker='x')

    # Decorate the plot
    plt.xlabel('x')
    plt.ylabel('y and {func_form}'.format(func_form=func_form))
    plt.title('Plot of the Functions and their Correlated Positive Sections')
    plt.annotate('Correlation: {corr}'.format(corr = corr[0]), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.legend()
    plt.grid(True)
    plt.show()



