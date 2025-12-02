import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats
import seaborn as sns

class eVTOLAnalyzer:
    def __init__(self, csv_pattern="evtol_simulation_results_*.csv"):
        """
        Initialize the analyzer with a pattern to match CSV files.
        
        Parameters:
        csv_pattern (str): Glob pattern to match CSV files
        """
        self.csv_pattern = csv_pattern
        self.data = None
        self.controllers = ['RLQRD', 'LQR', 'RLQR', 'Hinf', 'AEM']
        self.colors = {'RLQRD': 'blue', 'LQR': 'red', 'RLQR': 'black', 
                      'Hinf': 'green', 'AEM': 'magenta'}
        self.linestyles = {'RLQRD': '-', 'LQR': ':', 'RLQR': '--', 
                          'Hinf': '-.', 'AEM': '-'}
        
    def load_csv_files(self, directory="."):
        """
        Load all CSV files matching the pattern and combine them.
        
        Parameters:
        directory (str): Directory to search for CSV files
        
        Returns:
        dict: Dictionary containing combined data for each controller
        """
        csv_files = glob.glob(os.path.join(directory, self.csv_pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern: {self.csv_pattern}")
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {file}")
        
        # Initialize data structure
        combined_data = {}
        
        # Read all CSV files
        all_dataframes = []
        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            df['run_id'] = i  # Add run identifier
            all_dataframes.append(df)
        
        # Combine all dataframes
        full_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Organize data by controller and metric
        for controller in self.controllers:
            combined_data[controller] = {
                'state_norm': [],
                'input_norm': []
            }
        
        combined_data['disturbance_norm'] = []
        combined_data['time'] = None
        
        # Group by run_id and extract data
        for run_id in full_df['run_id'].unique():
            run_data = full_df[full_df['run_id'] == run_id].sort_values('time')
            
            # Store time (assuming same for all runs)
            if combined_data['time'] is None:
                combined_data['time'] = run_data['time'].values
            
            # Store controller data
            for controller in self.controllers:
                state_col = f'{controller}_state_norm'
                input_col = f'{controller}_input_norm'
                
                if state_col in run_data.columns:
                    combined_data[controller]['state_norm'].append(run_data[state_col].values)
                if input_col in run_data.columns:
                    combined_data[controller]['input_norm'].append(run_data[input_col].values)
            
            # Store disturbance data
            if 'disturbance_norm' in run_data.columns:
                combined_data['disturbance_norm'].append(run_data['disturbance_norm'].values)
        
        # Convert lists to numpy arrays
        for controller in self.controllers:
            if combined_data[controller]['state_norm']:
                combined_data[controller]['state_norm'] = np.array(combined_data[controller]['state_norm'])
            if combined_data[controller]['input_norm']:
                combined_data[controller]['input_norm'] = np.array(combined_data[controller]['input_norm'])
        
        if combined_data['disturbance_norm']:
            combined_data['disturbance_norm'] = np.array(combined_data['disturbance_norm'])
        
        self.data = combined_data
        return combined_data
    
    def compute_statistics(self, data_array):
        """
        Compute mean, standard deviation, and confidence intervals.
        
        Parameters:
        data_array (np.array): Array of shape (n_runs, n_timepoints)
        
        Returns:
        dict: Dictionary with statistical measures
        """
        if len(data_array) == 0:
            return None
        
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        
        # 95% confidence interval
        n_runs = data_array.shape[0]
        sem = std / np.sqrt(n_runs)  # Standard error of mean
        ci_95 = 1.96 * sem  # 95% confidence interval
        
        # Percentiles for robust bounds
        percentile_25 = np.percentile(data_array, 0, axis=0)
        percentile_75 = np.percentile(data_array, 75, axis=0)
        percentile_5 = np.percentile(data_array, 0, axis=0)
        percentile_95 = np.percentile(data_array, 100, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'upper_ci': mean + ci_95,
            'lower_ci': mean - ci_95,
            'upper_bound': mean + std,
            'lower_bound': mean - std,
            'percentile_25': percentile_25,
            'percentile_75': percentile_75,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95
        }
    
    def plot_state_norms_with_bounds(self, save_path=None, bound_type='ci'):
        """
        Plot state norms for all controllers with confidence bounds.
        
        Parameters:
        save_path (str): Path to save the plot
        bound_type (str): Type of bounds ('ci', 'std', 'percentile')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        plt.figure(figsize=(12, 8))
        time = self.data['time']
        
        for controller in self.controllers:
            state_data = self.data[controller]['state_norm']
            if len(state_data) == 0:
                continue
                
            stats = self.compute_statistics(state_data)
            if stats is None:
                continue
            
            color = self.colors[controller]
            linestyle = self.linestyles[controller]
            
            # Plot mean line
            plt.plot(time, stats['mean'], color=color, linestyle=linestyle, 
                    linewidth=2, label=f'{controller}', alpha=0.8)
            
            # Plot bounds based on type
            if bound_type == 'ci':
                upper = stats['upper_ci']
                lower = stats['lower_ci']
                bound_label = '95% CI'
            elif bound_type == 'std':
                upper = stats['upper_bound']
                lower = stats['lower_bound']
                bound_label = 'Mean ± Std'
            elif bound_type == 'percentile':
                upper = stats['percentile_75']
                lower = stats['percentile_25']
                bound_label = '25th-75th percentile'
            else:
                raise ValueError("bound_type must be 'ci', 'std', or 'percentile'")
            
            # Fill between bounds
            plt.fill_between(time, lower, upper, color=color, alpha=0.2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('State Norm $||x||_2$', fontsize=12)
        plt.title(f'State Norms with {bound_label}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State norms plot saved to: {save_path}")
        
        #plt.show()
    
    def plot_input_norms_with_bounds(self, save_path=None, bound_type='ci'):
        """
        Plot input norms for all controllers with confidence bounds.
        
        Parameters:
        save_path (str): Path to save the plot
        bound_type (str): Type of bounds ('ci', 'std', 'percentile')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        plt.figure(figsize=(12, 8))
        time = self.data['time']
        
        for controller in self.controllers:
            input_data = self.data[controller]['input_norm']
            if len(input_data) == 0:
                continue
                
            stats = self.compute_statistics(input_data)
            if stats is None:
                continue
            
            color = self.colors[controller]
            linestyle = self.linestyles[controller]
            
            # Plot mean line
            plt.plot(time, stats['mean'], color=color, linestyle=linestyle, 
                    linewidth=2, label=f'{controller}', alpha=0.8)
            
            # Plot bounds based on type
            if bound_type == 'ci':
                upper = stats['upper_ci']
                lower = stats['lower_ci']
                bound_label = '95% CI'
            elif bound_type == 'std':
                upper = stats['upper_bound']
                lower = stats['lower_bound']
                bound_label = 'Mean ± Std'
            elif bound_type == 'percentile':
                upper = stats['percentile_75']
                lower = stats['percentile_25']
                bound_label = '25th-75th percentile'
            
            # Fill between bounds
            plt.fill_between(time, lower, upper, color=color, alpha=0.2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Input Norm $||u||_2$', fontsize=12)
        plt.title(f'Input Norms with {bound_label}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Input norms plot saved to: {save_path}")
        
        #plt.show()
    
    def plot_combined_analysis(self, save_path=None, bound_type='ci'):
        """
        Create a combined plot with states, inputs, and disturbances.
        
        Parameters:
        save_path (str): Path to save the plot
        bound_type (str): Type of bounds ('ci', 'std', 'percentile')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        time = self.data['time']
        
        # Determine bound label
        bound_labels = {
            'ci': '95% CI',
            'std': 'Mean ± Std',
            'percentile': '25th-75th percentile'
        }
        bound_label = bound_labels.get(bound_type, '95% CI')
        
        # Plot 1: State norms
        ax1 = axes[0]
        for controller in self.controllers:
            state_data = self.data[controller]['state_norm']
            if len(state_data) == 0:
                continue
                
            stats = self.compute_statistics(state_data)
            if stats is None:
                continue
            
            color = self.colors[controller]
            linestyle = self.linestyles[controller]
            
            ax1.plot(time, stats['mean'], color=color, linestyle=linestyle, 
                    linewidth=1, label=f'{controller}', alpha=0.8)
            
            # Select bounds
            if bound_type == 'ci':
                upper, lower = stats['upper_ci'], stats['lower_ci']
            elif bound_type == 'std':
                upper, lower = stats['upper_bound'], stats['lower_bound']
            else:  # percentile
                upper, lower = stats['percentile_75'], stats['percentile_25']
            
            ax1.fill_between(time, lower, upper, color=color, alpha=0.2)
        
        ax1.set_ylabel('State Norm $||x||_2$', fontsize=12)
        ax1.set_title(f'State Norms', fontsize=14)
        ax1.set_ylim(bottom=0, top=0.30)
        ax1.grid(True)
        #ax1.legend(fontsize=12)
        # make ticks fontsize 14
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # Plot 2: Input norms
        ax2 = axes[1]
        for controller in self.controllers:
            input_data = self.data[controller]['input_norm']
            if len(input_data) == 0:
                continue
                
            stats = self.compute_statistics(input_data)
            if stats is None:
                continue
            
            color = self.colors[controller]
            linestyle = self.linestyles[controller]
            
            ax2.plot(time, stats['mean'], color=color, linestyle=linestyle, 
                    linewidth=1, label=f'{controller}', alpha=0.8)
            
            # Select bounds
            if bound_type == 'ci':
                upper, lower = stats['upper_ci'], stats['lower_ci']
            elif bound_type == 'std':
                upper, lower = stats['upper_bound'], stats['lower_bound']
            else:  # percentile
                upper, lower = stats['percentile_75'], stats['percentile_25']
            
            ax2.fill_between(time, lower, upper, color=color, alpha=0.2)
        
        ax2.set_ylabel('Input Norm $||u||_2$', fontsize=12)
        ax2.set_title(f'Input Norms', fontsize=14)
        ax2.set_ylim(bottom=0, top=52)
        ax2.grid(True)
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=14)

        # Plot 3: Disturbances
        ax3 = axes[2]
        disturbance_data = self.data['disturbance_norm']
        if len(disturbance_data) > 0:
            stats = self.compute_statistics(disturbance_data)
            if stats is not None:
                ax3.plot(time, 100*stats['mean'], color='black', linewidth=1, 
                        label='Disturbance', alpha=0.8)
                
                # Select bounds
                if bound_type == 'ci':
                    upper, lower = stats['upper_ci'], stats['lower_ci']
                elif bound_type == 'std':
                    upper, lower = stats['upper_bound'], stats['lower_bound']
                else:  # percentile
                    upper, lower = stats['percentile_75'], stats['percentile_25']
                
                ax3.fill_between(time, lower, upper, color='black', alpha=0.2)
        
        ax3.set_xlabel('Time (s)', fontsize=14)
        ax3.set_ylabel('Disturbance Norm $||w||_2$', fontsize=12)
        ax3.set_title(f'Disturbances', fontsize=14)
        ax3.grid(True)
        ax3.legend(fontsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined analysis plot saved to: {save_path}")
        
        #plt.show()
    
    def compute_mse_statistics(self):
        """
        Compute MSE statistics for each controller across all runs.
        
        Returns:
        dict: MSE statistics for each controller
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        mse_stats = {}
        
        for controller in self.controllers:
            state_data = self.data[controller]['state_norm']
            if len(state_data) == 0:
                continue
            
            # Compute MSE for each run
            mse_values = []
            for run_data in state_data:
                mse = np.mean(run_data**2)
                mse_values.append(mse)
            
            mse_values = np.array(mse_values)
            
            mse_stats[controller] = {
                'mean_mse': np.mean(mse_values),
                'std_mse': np.std(mse_values),
                'min_mse': np.min(mse_values),
                'max_mse': np.max(mse_values),
                'all_mse': mse_values
            }
        
        return mse_stats
    
    def plot_mse_comparison(self, save_path=None):
        """
        Create a bar plot comparing MSE values across controllers.
        
        Parameters:
        save_path (str): Path to save the plot
        """
        mse_stats = self.compute_mse_statistics()
        
        controllers = list(mse_stats.keys())
        mean_mse = [mse_stats[ctrl]['mean_mse'] for ctrl in controllers]
        std_mse = [mse_stats[ctrl]['std_mse'] for ctrl in controllers]
        colors = [self.colors[ctrl] for ctrl in controllers]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(controllers, mean_mse, yerr=std_mse, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title('MSE Comparison Across Controllers', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mse, std) in enumerate(zip(bars, mean_mse, std_mse)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01*max(mean_mse),
                    f'{mse:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MSE comparison plot saved to: {save_path}")
        
        #plt.show()
        
        return mse_stats
    
    def plot_3d_depth_plot(self, save_path=None):
        """
        Create a 3D depth plot similar to the style in evtol_dynamics.py.
        Each controller is plotted at a different depth level.
        
        Parameters:
        save_path (str): Path to save the plot
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        time = self.data['time']
        
        # Prepare data series - compute mean across all runs for each controller
        state_series = []
        state_MSE = []
        
        # Define controller order to match the evtol_dynamics.py style
        controller_order = ['AEM', 'RLQRD', 'RLQR', 'LQR', 'Hinf']
        
        for controller in controller_order:
            state_data = self.data[controller]['state_norm']
            if len(state_data) > 0:
                # Compute statistics
                stats = self.compute_statistics(state_data)
                if stats is not None:
                    # Use mean trajectory
                    mean_trajectory = stats['mean']
                    state_series.append(mean_trajectory)
                    
                    # Compute MSE
                    mse = np.mean(stats['mean']**2)
                    state_MSE.append(mse)
                else:
                    # Fallback to zeros if no data
                    state_series.append(np.zeros_like(time))
                    state_MSE.append(0.0)
            else:
                # Fallback to zeros if no data
                state_series.append(np.zeros_like(time))
                state_MSE.append(0.0)
        
        print("State MSEs: \n")
        mse_string = ", ".join([f"{controller}: {mse:.8f}" if mse < 1e-3 else f"{controller}: {mse:.4f}" 
                               for controller, mse in zip(controller_order, state_MSE)])
        print(mse_string)
        
        # Depth levels and labels (matching evtol_dynamics.py style)
        depths = [0, 1, 2, 3, 4]
        labels = ['AEM', 'RLQRD', 'RLQR', 'LQR', r'$H_{\infty}$']
        colors = ['m', 'b', 'k', 'r', 'g']
        linestyles = ['-', '-', '-', '-', '-']
        
        # Plot lines with depth
        for depth, series, label, color, ls in zip(depths, state_series, labels, colors, linestyles):
            ax.plot(time, series, zs=depth, zdir='y', label=label, color=color, linestyle=ls, linewidth=2)
        
        # Label settings
        ax.set_xlabel('Time (s)', labelpad=12)
        ax.set_zlabel(r'$||x||_{2}$', labelpad=12)
        ax.set_zlim(0, 5)
        ax.set_yticks(depths)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title('State Norms over Time with Depth Plot (Mean Trajectories)', pad=20)
        
        # Aspect ratio
        ax.set_box_aspect([12, 6, 6])
        
        # CLEANUP: remove background panes and grid (matching evtol_dynamics.py style)
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('k')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = True
        
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.legend(loc='upper left', fontsize=12, frameon=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D depth plot saved to: {save_path}")
        
        plt.show()
    
    def plot_3d_depth_plot_with_bounds(self, save_path=None, bound_type='percentile'):
        """
        Create a 3D depth plot with confidence bounds using fill_between effect.
        Each controller is plotted at a different depth level with shaded confidence regions.
        
        Parameters:
        save_path (str): Path to save the plot
        bound_type (str): Type of bounds ('ci', 'std', 'percentile')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        time = self.data['time']
        
        # Define controller order to match the evtol_dynamics.py style
        controller_order = ['AEM', 'RLQRD', 'RLQR', 'LQR', 'Hinf']
        
        # Depth levels and labels (matching evtol_dynamics.py style)
        depths = [0, 1, 2, 3, 4]
        labels = ['AEM', 'RLQRD', 'RLQR', 'LQR', r'$H_{\infty}$']
        colors = ['m', 'b', 'k', 'r', 'g']
        linestyles = ['-', '-', '-', '-', '-']
        
        # Plot lines with depth and bounds
        for i, controller in enumerate(controller_order):
            state_data = self.data[controller]['state_norm']
            if len(state_data) > 0:
                # Compute statistics
                stats = self.compute_statistics(state_data)
                if stats is not None:
                    depth = depths[i]
                    label = labels[i]
                    color = colors[i]
                    ls = linestyles[i]
                    
                    # Plot mean line
                    ax.plot(time, stats['mean'], zs=depth, zdir='y', label=label, 
                           color=color, linestyle=ls, linewidth=2)
                    
                    # Select bounds based on type
                    if bound_type == 'ci':
                        upper = stats['upper_ci']
                        lower = stats['lower_ci']
                    elif bound_type == 'std':
                        upper = stats['upper_bound']
                        lower = stats['lower_bound']
                    else:  # percentile
                        upper = stats['percentile_75']
                        lower = stats['percentile_25']
                    
                    # Create 3D fill_between effect using polygons
                    # Sample points to reduce polygon complexity
                    sample_indices = np.arange(0, len(time), max(1, len(time)//100))
                    time_sampled = time[sample_indices]
                    upper_sampled = upper[sample_indices]
                    lower_sampled = lower[sample_indices]
                    
                    # Create vertices for the filled region
                    vertices = []
                    for j in range(len(time_sampled) - 1):
                        # Create a quad between adjacent time points
                        quad = [
                            [time_sampled[j], depth, lower_sampled[j]],
                            [time_sampled[j], depth, upper_sampled[j]],
                            [time_sampled[j+1], depth, upper_sampled[j+1]],
                            [time_sampled[j+1], depth, lower_sampled[j+1]]
                        ]
                        vertices.append(quad)
                    
                    # Add the filled regions as polygons
                    if vertices:
                        poly3d = Poly3DCollection(vertices, alpha=0.5, facecolors=color, edgecolors='none')
                        ax.add_collection3d(poly3d)
        
        # Label settings
        ax.set_xlabel('Time (s)', labelpad=12)
        ax.set_zlabel(r'$||x||_{2}$', labelpad=12)
        ax.set_zlim(0, 5)
        ax.set_yticks(depths)
        ax.set_yticklabels(labels, fontsize=10)
        
        bound_labels = {
            'ci': '95% CI',
            'std': 'Mean ± Std',
            'percentile': '25th-75th percentile'
        }
        bound_label = bound_labels.get(bound_type, '95% CI')
        ax.set_title(f'State Norms over Time with Depth Plot ({bound_label})', pad=20)
        
        # Aspect ratio
        ax.set_box_aspect([12, 6, 6])
        
        # CLEANUP: remove background panes and grid (matching evtol_dynamics.py style)
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('k')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = True
        
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.legend(loc='upper left', fontsize=12, frameon=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D depth plot with bounds saved to: {save_path}")
        
        plt.show()
    
    def plot_2d_depth_plot_with_fill(self, save_path=None, bound_type='percentile'):
        """
        Create a 2D plot with offset controllers and fill_between for confidence bounds.
        This is an alternative to 3D that uses traditional fill_between.
        
        Parameters:
        save_path (str): Path to save the plot
        bound_type (str): Type of bounds ('ci', 'std', 'percentile')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        time = self.data['time']
        
        # Define controller order and offsets
        controller_order = ['AEM', 'RLQRD', 'RLQR', 'LQR', 'Hinf']
        labels = ['AEM', 'RLQRD', 'RLQR', 'LQR', r'$H_{\infty}$']
        colors = ['m', 'b', 'k', 'r', 'g']
        linestyles = ['-', '-', '-', '-', '-']
        offsets = [0, 0.5, 1.0, 1.5, 2.0]  # Vertical offsets to separate controllers
        
        # Plot lines with offsets and fill_between
        for i, controller in enumerate(controller_order):
            state_data = self.data[controller]['state_norm']
            if len(state_data) > 0:
                # Compute statistics
                stats = self.compute_statistics(state_data)
                if stats is not None:
                    offset = offsets[i]
                    label = labels[i]
                    color = colors[i]
                    ls = linestyles[i]
                    
                    # Apply offset to all values
                    mean_offset = stats['mean'] + offset
                    
                    # Select bounds based on type
                    if bound_type == 'ci':
                        upper = stats['upper_ci'] + offset
                        lower = stats['lower_ci'] + offset
                    elif bound_type == 'std':
                        upper = stats['upper_bound'] + offset
                        lower = stats['lower_bound'] + offset
                    else:  # percentile
                        upper = stats['percentile_75'] + offset
                        lower = stats['percentile_25'] + offset
                    
                    # Plot mean line
                    plt.plot(time, mean_offset, color=color, linestyle=ls, 
                            linewidth=2, label=label, alpha=0.8)
                    
                    # Fill between bounds
                    plt.fill_between(time, lower, upper, color=color, alpha=0.2)
        
        # Label settings
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('State Norm $||x||_2$ (with offsets)', fontsize=12)
        
        bound_labels = {
            'ci': '95% CI',
            'std': 'Mean ± Std',
            'percentile': '25th-75th percentile'
        }
        bound_label = bound_labels.get(bound_type, '95% CI')
        plt.title(f'State Norms over Time with Offsets ({bound_label})', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D depth plot with fill_between saved to: {save_path}")
        
        plt.show()


def main():
    """
    Example usage of the eVTOLAnalyzer class.
    """
    # Initialize analyzer
    analyzer = eVTOLAnalyzer("evtol_simulation_results_*.csv")
    
    try:
        # Load CSV files
        data = analyzer.load_csv_files(".")
        print(f"\nLoaded data for {len(data['time'])} time points")
        
        # Create plots with different bound types
        print("\nCreating plots...")
        
        ##State norms with 95% confidence intervals
        analyzer.plot_state_norms_with_bounds(
            save_path="state_norms_ci.png", 
            bound_type='ci'
        )
        
        # # Input norms with standard deviation bounds
        # analyzer.plot_input_norms_with_bounds(
        #     save_path="input_norms_std.png", 
        #     bound_type='std'
        # )
        
        # Combined analysis with percentile bounds
        analyzer.plot_combined_analysis(
            save_path="combined_analysis_percentile.png", 
            bound_type='percentile'
        )
        
        # # 3D depth plot (mean trajectories only)
        # analyzer.plot_3d_depth_plot(save_path="3d_depth_plot.png")
        
        # 3D depth plot with bounds (using Poly3DCollection for fill effect)
        analyzer.plot_3d_depth_plot_with_bounds(
            save_path="3d_depth_plot_bounds.png", 
            bound_type='percentile'
        )
        
        # # 2D depth plot with traditional fill_between
        # analyzer.plot_2d_depth_plot_with_fill(
        #     save_path="2d_depth_plot_fill.png", 
        #     bound_type='percentile'
        # )
        
        # # MSE comparison
        # mse_stats = analyzer.plot_mse_comparison(save_path="mse_comparison.png")
        
        # Print MSE statistics
        # print("\nMSE Statistics:")
        # print("-" * 50)
        # for controller, stats in mse_stats.items():
        #     print(f"{controller}:")
        #     print(f"  Mean MSE: {stats['mean_mse']:.6f} ± {stats['std_mse']:.6f}")
        #     print(f"  Range: [{stats['min_mse']:.6f}, {stats['max_mse']:.6f}]")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have CSV files matching the pattern 'evtol_simulation_results_*.csv'")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    plt.show()
