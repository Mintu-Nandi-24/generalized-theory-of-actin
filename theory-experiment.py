import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams



##------------------------------------------------------------------------------------------------------
# Bootstrap error bar calculation
##------------------------------------------------------------------------------------------------------

def calculate_bootstrap_error_bars(data, n_bootstrap=5000, ci_width=95):
    n_samples, n_features = data.shape
    bootstrap_means = np.full((n_bootstrap, n_features), np.nan)
    bootstrap_vars = np.full((n_bootstrap, n_features), np.nan)
    
    # Calculate percentiles for CI
    lower_percentile = (100 - ci_width)/2
    upper_percentile = 100 - lower_percentile
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resampled_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        resampled_data = data[resampled_indices]
        
        for j in range(n_features):
            valid_values = resampled_data[:, j][~np.isnan(resampled_data[:, j])]
            
            if len(valid_values) > 1:
                bootstrap_means[i, j] = np.mean(valid_values)
                bootstrap_vars[i, j] = np.var(valid_values, ddof=1)
            elif len(valid_values) == 1:
                bootstrap_means[i, j] = valid_values[0]
                bootstrap_vars[i, j] = 0.0  # No variance for single point
    
    # Protected Fano ratio calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        fano_ratios = np.divide(bootstrap_vars, bootstrap_means)
        fano_ratios[~np.isfinite(fano_ratios)] = np.nan
    
    # Calculate confidence intervals
    def safe_ci(arr):
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            lower = np.percentile(valid, lower_percentile)
            upper = np.percentile(valid, upper_percentile)
            return (upper - lower)/2
        return np.nan
    
    mean_error_bars = np.array([safe_ci(bootstrap_means[:, j]) for j in range(n_features)])
    variance_error_bars = np.array([safe_ci(bootstrap_vars[:, j]) for j in range(n_features)])
    fano_error_bars = np.array([safe_ci(fano_ratios[:, j]) for j in range(n_features)])
    
    return mean_error_bars, variance_error_bars, fano_error_bars


##------------------------------------------------------------------------------------------------------
# Experimental data analysis
##------------------------------------------------------------------------------------------------------

# Function to interpolate data for a given time window
def interpolate_data(time, length, time_window):
    interpolated_length = np.interp(time_window, time, length)
    return interpolated_length


# Define filenames
filenames = [f"fil1-{i}.txt" for i in range(1, 37)]

# Initialize list to store final times
final_times = []

for fname in filenames:
    # Load time column (assumes time is in the first column)
    data = np.loadtxt(fname)
    time_column = data[:, 0]  # adjust index if time is in another column
    final_times.append(time_column[-1])  # take the last time point

# Find the minimum of the final times
final_time = min(final_times)

# Define the time window parameters
start_time = 0 
end_time = final_time 
nvalue = 20

interval = (end_time - start_time) / nvalue
time_window = np.arange(start_time, end_time, interval)


# Initialize an array to store interpolated lengths for all files
dlengths = []

# Read data from files, interpolate for each file, and save to separate files
for filename in filenames:
    # Load data from file
    data = np.loadtxt(filename)
    time = data[:, 0]  # Extract time column
    length = data[:, 1]  # Extract length column
    
    # Interpolate data for the time window
    interpolated_length = interpolate_data(time, length, time_window)
    
    # Append adjusted interpolated length to the array for all files
    dlengths.append(interpolated_length)   
    
# Convert dlengths to a numpy array
dlengths = np.array(dlengths)


# Calculate the moments of length at each time point across all files
mean_dl = np.mean(dlengths, axis=0)
var_dl = np.var(dlengths, axis=0, ddof=1)

# Initialize Fano array
fano_dl = np.zeros_like(mean_dl)

# Avoid division by zero
nonzero_mask = mean_dl != 0
fano_dl[nonzero_mask] = var_dl[nonzero_mask] / mean_dl[nonzero_mask]
    
# Calculate error bars for each metric
mean_error_bars, variance_error_bars, fano_factor_error_bars = calculate_bootstrap_error_bars(dlengths)

##------------------------------------------------------------------------------------------------------
# Theoretical calculations
##------------------------------------------------------------------------------------------------------

# ------------------------
# Theoretical Data Processing
# ------------------------
# Parameters
R = 16 #21 /1.7 #11 
r = 6 #7 /1.7 #1.8 
k = 14.7*(10**(-5)) #8.1*(10**(-5)) *70*0.4 *0.65
kfp = 8.8*(10**(-3)) #29.1*(10**(-3)) *0.78 *0.7 /1.8
F = 5
kf = kfp * F
Tf = final_time-15
dt = 1
time_values = np.arange(0.01, Tf, dt)

# ------------------------
# Theoretical Calculation
# ------------------------
DF = k + kf  # total dissociation
AF = k / DF  # fraction of time in free state

mean_theory = []
dmu_dR_list = []
var_intrinsic_list = []

for t in time_values:
    exp_DFt = np.exp(-DF * t)
    exp_2DFt = np.exp(-2 * DF * t)

    # Calculate mean
    mean = (
        r * AF * t + R * (1 - AF) * t +
        ((r - R) * (1 - AF) / DF) * (1 - exp_DFt)
    )
    mean_theory.append(mean)

    # Calculate intrinsic variance
    term1 = ((r - R) * (1 - AF) / DF**2) * (DF + (r - R) * (1 - 5 * AF))
    term2 = (
        R + AF * (r - R) + ((2 * AF * (1 - AF) * (r - R)**2) / DF)
    ) * t
    term3 = -(((1 - AF) * (r - R)) / DF)**2 * exp_2DFt
    term4 = (((1 - AF) * (r - R)) / DF**2) * (4 * AF * (r - R) -
             (DF * (1 + 2 * (1 - 2 * AF) * (r - R) * t))) * exp_DFt

    var_intrinsic = term1 + term2 + term3 + term4
    var_intrinsic_list.append(var_intrinsic)

# Convert to arrays
mean_theory = np.array(mean_theory)
var_intrinsic_list = np.array(var_intrinsic_list)
fano_intrinsic_list = var_intrinsic_list/mean_theory



# Plot aesthetics
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20  # or any larger value
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
rcParams['axes.linewidth'] = 1.5
rcParams['lines.linewidth'] = 2.5
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.5
rcParams['legend.frameon'] = False

plt.figure(figsize=(8, 6))
plt.plot(time_values, fano_intrinsic_list, '-', color='tab:green', label='Theory')
plt.errorbar(time_window, fano_dl, yerr=fano_factor_error_bars, fmt='o',
             color='tab:blue', markersize=6, elinewidth=1.5, capsize=5, 
             markerfacecolor='none', markeredgewidth=1.5, label='Experiment')
plt.xlabel('Time (s)')
plt.ylabel('Fano factor')
plt.xlim(-5,305)
plt.legend(loc='upper right')
plt.savefig("Figure5.pdf", format="pdf", bbox_inches="tight")

plt.show()


plt.figure(figsize=(8, 6))
plt.plot(time_values, mean_theory*0.0027, '-', color='tab:green', label='Theory')
plt.errorbar(time_window, mean_dl*0.0027, yerr=mean_error_bars*0.0027, fmt='o',
             color='tab:blue', markersize=6, elinewidth=1.5, capsize=5, 
             markerfacecolor='none', markeredgewidth=1.5, label='Experiment')
plt.xlabel('Time (s)')
plt.ylabel('Mean growth')
plt.xlim(-5,305)
plt.legend(loc='upper left')

plt.show()


# Plot aesthetics
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20  # or any larger value
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
rcParams['axes.linewidth'] = 1.5
rcParams['lines.linewidth'] = 2.5
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.5
rcParams['legend.frameon'] = False

plt.figure(figsize=(8, 6))

# Create a colormap that fades from dark gray to light gray
colors = plt.cm.Greys(np.linspace(0.4, 0.8, len(filenames)))

# Plot each filament's trajectory
for i, filename in enumerate(filenames):
    data = np.loadtxt(filename)
    time = data[:, 0]
    length = data[:, 1]
    plt.plot(time, length*0.0027, color=colors[i], alpha=0.7, linewidth=1.5)

# Add labels and adjust limits
plt.xlabel('Time (s)')
plt.ylabel('Length (μm)')
plt.xlim(0, final_time)
plt.ylim(bottom=0)  # Start y-axis at 0

plt.tight_layout()
plt.savefig("traj.pdf", format="pdf", bbox_inches="tight")
plt.show()



# # Plot aesthetics
# rcParams['font.family'] = 'Arial'
# rcParams['font.size'] = 20  # or any larger value
# rcParams['axes.titlesize'] = 20
# rcParams['axes.labelsize'] = 20
# rcParams['xtick.labelsize'] = 20
# rcParams['ytick.labelsize'] = 20
# rcParams['legend.fontsize'] = 20
# rcParams['axes.linewidth'] = 1.5
# rcParams['lines.linewidth'] = 2
# rcParams['xtick.major.size'] = 6
# rcParams['xtick.major.width'] = 1.5
# rcParams['ytick.major.size'] = 6
# rcParams['ytick.major.width'] = 1.5
# rcParams['legend.frameon'] = False

# # Calculate upper and lower bounds for the error band
# fano_upper = fano_dl + fano_factor_error_bars
# fano_lower = fano_dl - fano_factor_error_bars

# # Create the plot
# plt.figure(figsize=(8, 6))

# # Plot the theoretical curve
# plt.plot(time_values, fano_intrinsic_list, '-', color='tab:green', label='Theory')

# # Plot the experimental data with error band
# plt.plot(time_window, fano_dl, '-', color='tab:blue', markersize=6, label='Experiment')
# plt.fill_between(time_window, fano_lower, fano_upper, color='tab:blue', alpha=0.2)

# # Formatting
# plt.xlabel('Time (s)')
# plt.ylabel('Fano factor')
# plt.xlim(-5, 305)
# plt.legend(loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.5)

# # Save and show
# plt.savefig("FFigure5.pdf", format="pdf", bbox_inches="tight", dpi=300)
# plt.show()