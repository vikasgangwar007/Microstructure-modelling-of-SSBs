import numpy as np
import os
import glob
from multiprocessing import Pool

import pandas as pd
from structure_generation.structgen import generate_am_structure, generate_se_structure, v_frac_after_combining
from percolation_analysis.percolating_clusters import (
    extract_se_percolating_clusters_y_range,
    find_percolating_se_particles,
    extract_am_percolating_clusters_y_range,
    find_percolating_am_particles
)
from percolation_analysis.output_parameters import (
    calculate_utilization_level,
    calculate_utilization_level_se,
    specific_surface_area_with_overlaps,
    active_interface_area3
)
from excel_saving_handler.excel_handler import get_workbook_and_sheet, save_data_to_excel

# Define the main Excel file for final data collection and a directory for temporary outputs.
excel_filename = 'Simulation_R1.xlsx'
temp_output_dir = 'temp_excel_outputs_R1'

# Active Material (AM) volume fractions to be simulated.
am_fractions = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58,  0.60, 0.62,0.64, 0.66, 0.68, 0.70]

# AM diameters to sweep (in µm).
am_diameters = [5, 6]

# Default AM particle parameters (will be overridden per diameter).
am_diameter = 5
am_mean_radius = am_diameter / 2  # Mean AM particle radius (μm).
overlap_fraction_am = 0.80

# Solid Electrolyte (SE) particle parameters (fixed).
se_diameter = 3
se_mean_radius = se_diameter / 2  # Mean SE particle radius (μm).
overlap_fraction_se = 0.90  # Scale for overlap determination.

# Microstructure porosity.
porosity = 0.2

# Microstructure dimensions (x, y, z).
size = (25,50,25)

# Other simulation parameters.
max_radius = max(am_mean_radius, se_mean_radius)
max_range = 2 * max_radius
num_points = 1000  # Points for surface area calculations.
voxel_resolution = 0.2  # Voxel side length.
SEED = 4  # Random seed.

# Additional parameters for fixed radius distribution.
deviation_fraction = 0
am_std_radius = 0
se_std_radius = 0

# Thresholds for considering AM and SE fractions.
am_fraction_threshold = [0.0001]
se_fraction_threshold = [0.0001]

# Interval for saving particle coordinates to Excel.
PARTICLE_SAVE_INTERVAL = 1000

# Compile initial simulation parameters into a dictionary for logging.
common_simulation_parameters = {
    'AM Mean Radius': am_mean_radius,
    'Overlap Fraction AM': overlap_fraction_am,
    'SE Mean Radius': se_mean_radius,
    'Overlap Fraction SE': overlap_fraction_se,
    'Deviation Fraction': deviation_fraction,
    'AM Standard Deviation Radius': am_std_radius,
    'SE Standard Deviation Radius': se_std_radius,
    'Max Range': max_range,
    'Number of Points': num_points,
    'Microstructure Size': size,
    'porosity': porosity,
    'Voxel Resolution': voxel_resolution,
    'SEED': SEED,
    'AM fraction threshold': am_fraction_threshold,
    'SE fraction threshold': se_fraction_threshold,
    'Particle Save Interval': PARTICLE_SAVE_INTERVAL,
    'Temp Output Dir': temp_output_dir
}

# Convert initial parameters to DataFrame (for 'Initial Parameters' sheet if needed).
initial_params_df = pd.DataFrame([common_simulation_parameters])

def work2(params_tuple):
    """
    Executes a single simulation run for a given AM fraction and diameter,
    generating microstructure, performing percolation analysis, and calculating key metrics.
    """
    am_fraction, am_diameter, common_params = params_tuple

    # Set AM mean radius based on diameter for this run.
    am_mean_radius = am_diameter / 2
    overlap_fraction_am = common_params['Overlap Fraction AM']
    se_mean_radius = common_params['SE Mean Radius']
    overlap_fraction_se = common_params['Overlap Fraction SE']
    max_range = common_params['Max Range']
    num_points = common_params['Number of Points']
    size = common_params['Microstructure Size']
    porosity = common_params['porosity']
    voxel_resolution = common_params['Voxel Resolution']
    SEED = common_params['SEED']
    am_fraction_threshold = common_params['AM fraction threshold']
    se_fraction_threshold = common_params['SE fraction threshold']
    particle_save_interval = common_params['Particle Save Interval']
    temp_output_dir = common_params['Temp Output Dir']

    # Initialize output metrics.
    ul, ul_se, sa, sa2, aia, aia2, aia3 = 0, 0, 0, 0, 0, 0, 0

    # Compute SE fraction from porosity and AM fraction.
    se_fraction = 1 - porosity - am_fraction
    elapsed_time_am = 0
    elapsed_time_se = 0

    # Path for temporary Excel file for this run (include diameter).
    temp_excel_path = os.path.join(
        temp_output_dir,
        f'{size}_temp_data_AM_{am_fraction:.2f}_{int(am_diameter)}um.xlsx'
    )

    # Voxel grid initialization.
    voxel_dimensions = np.ceil(np.array(size) / voxel_resolution).astype(int)
    voxel_grid = np.zeros(voxel_dimensions, dtype=int)

    # --- New Log: Starting simulation info with AM and SE fractions and diameter ---
    print(f'Starting simulation: Size={size}, AM Diameter={am_diameter}um, AM Fraction={am_fraction:.2f}, SE Fraction={se_fraction:.2f}')

    # --- Microstructure Generation ---
    # Generate AM structure.
    voxel_grid, am_coordinates, elapsed_time_am = generate_am_structure(
        am_mean_radius, size, am_fraction, max_range, overlap_fraction_am, SEED,
        am_fraction_threshold, voxel_resolution, voxel_dimensions, temp_excel_path, particle_save_interval
    )
    # Generate SE structure on top of AM.
    voxel_grid, se_coordinates, elapsed_time_se = generate_se_structure(
        se_mean_radius, size, se_fraction, voxel_grid, max_range, overlap_fraction_se,
        am_coordinates, SEED, se_fraction_threshold, am_fraction,
        voxel_resolution, voxel_dimensions, temp_excel_path, particle_save_interval
    )

    # --- Percolation Analysis and Metric Calculation ---
    # Actual volume fractions after combining.
    vf_am_AC, vf_se_AC = v_frac_after_combining(voxel_grid, size, voxel_dimensions)

    # Identify percolating clusters along y-direction.
    percolating_am_clusters_yrange, num_am_perc, am_cluster_sizes_yrange = extract_am_percolating_clusters_y_range(
        voxel_grid, size, voxel_resolution, am_fraction
    )
    percolating_se_clusters_yrange, num_se_perc, se_cluster_sizes_yrange = extract_se_percolating_clusters_y_range(
        voxel_grid, size, voxel_resolution, am_fraction
    )

    # Coordinates of percolating particles.
    percolating_am_coordinates, _ = find_percolating_am_particles(
        am_coordinates, percolating_am_clusters_yrange, am_fraction, size, voxel_resolution
    )
    percolating_se_coordinates, _ = find_percolating_se_particles(
        se_coordinates, percolating_se_clusters_yrange, am_fraction, size, voxel_resolution
    )

    # Compute utilization and interface metrics.
    if num_am_perc:
        ul = calculate_utilization_level(am_cluster_sizes_yrange, voxel_grid)
        sa = 1
        sa2 = specific_surface_area_with_overlaps(percolating_am_coordinates, num_points, max_range, am_fraction)
        aia = 1
        aia2 = 1
    if num_se_perc:
        ul_se = calculate_utilization_level_se(se_cluster_sizes_yrange, voxel_grid)
    if num_am_perc and num_se_perc:
        aia3 = active_interface_area3(
            percolating_am_coordinates, percolating_se_coordinates,
            num_points, max_range, am_fraction
        )

    # Compile results for this run.
    result_dict = {
        'Size': size,
        'AM Diameter': am_diameter,
        'AM Fraction': am_fraction,
        'SE Fraction': se_fraction,
        'vf_am_AC': vf_am_AC,
        'vf_se_AC': vf_se_AC,
        'UL': ul,
        'UL_se': ul_se,
        'SA': sa,
        'SA2': sa2,
        'AIA': aia,
        'AIA2': aia2,
        'AIA3': aia3,
        'AM Gen time(s)': elapsed_time_am,
        'SE Gen time(s)': elapsed_time_se,
        'Temp_Excel_Path': temp_excel_path  # Temp file path for copying data later.
    }

    # Save run summary to temp Excel.
    try:
        temp_results_df = pd.DataFrame([result_dict])
        # Exclude the Temp_Excel_Path column when writing run summary.
        temp_results_df = temp_results_df.drop(columns=['Temp_Excel_Path'], errors='ignore')
        workbook_summary, sheet_summary = get_workbook_and_sheet(
            temp_excel_path, 'Run_Summary', list(temp_results_df.columns)
        )
        save_data_to_excel(workbook_summary, sheet_summary, temp_results_df.values.tolist(), temp_excel_path)
    except Exception as e:
        print(f"Error saving run summary to temp file {temp_excel_path}: {e}")

    # --- New Log: Completed simulation info with AM and SE fractions and diameter ---
    print(f'Completed simulation: Size={size}, AM Diameter={am_diameter}um, AM Fraction={am_fraction:.2f}, SE Fraction={se_fraction:.2f}')
    return result_dict

if __name__ == "__main__":
    # Ensure temp output directory exists and clean it.
    os.makedirs(temp_output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(temp_output_dir, '*_temp_data_AM_*.xlsx')):
        os.remove(f)

    # Initialize main Excel with initial parameters (optional).
    workbook_initial, sheet_initial = get_workbook_and_sheet(
        excel_filename, 'Initial Parameters', list(initial_params_df.columns)
    )
    save_data_to_excel(workbook_initial, sheet_initial, initial_params_df.values.tolist(), excel_filename)

    # Prepare arguments for multiprocessing: each (AM fraction, diameter) pair.
    pool_args = []
    for diam in am_diameters:
        # Copy common params and update AM radius if needed.
        params_copy = common_simulation_parameters.copy()
        params_copy['AM Mean Radius'] = diam / 2
        for af in am_fractions:
            pool_args.append((af, diam, params_copy))

    # Run simulations in parallel for all combinations.
    with Pool() as pool:
        results = pool.map(work2, pool_args)

    # Consolidate results into the main Excel file summary.
    final_workbook, _ = get_workbook_and_sheet(excel_filename, 'Results')

    # Create DataFrame from results (exclude Temp_Excel_Path).
    results_df_summary = pd.DataFrame([
        {k: v for k, v in res.items() if k not in ('Temp_Excel_Path')} for res in results
    ], columns=[
        'Size', 'AM Diameter', 'AM Fraction', 'SE Fraction', 'vf_am_AC', 'vf_se_AC',
        'UL', 'UL_se', 'SA', 'SA2', 'AIA', 'AIA2', 'AIA3',
        'AM Gen time(s)', 'SE Gen time(s)'
    ])

    # Save the summary results.
    workbook_results, sheet_results = get_workbook_and_sheet(
        excel_filename, 'Results', list(results_df_summary.columns)
    )
    save_data_to_excel(workbook_results, sheet_results, results_df_summary.values.tolist(), excel_filename)

    # Copy detailed particle coordinate data to final Excel.
    with pd.ExcelWriter(excel_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        for res in results:
            am_fraction = res['AM Fraction']
            am_diameter = res['AM Diameter']
            temp_excel_path = res['Temp_Excel_Path']
            # Copy AM sheet if exists.
            try:
                df_am = pd.read_excel(temp_excel_path, sheet_name=f'AM_{am_fraction:.2f}')
                df_am.to_excel(writer, sheet_name=f'AM_{am_fraction:.2f}_{int(am_diameter)}um', index=False)
            except Exception as e:
                print(f"Warning: could not copy AM sheet for AM_frac {am_fraction:.2f}, diameter {am_diameter}: {e}")
            # Copy SE sheet if exists.
            try:
                df_se = pd.read_excel(temp_excel_path, sheet_name=f'SE_{am_fraction:.2f}')
                df_se.to_excel(writer, sheet_name=f'SE_{am_fraction:.2f}_{int(am_diameter)}um', index=False)
            except Exception as e:
                print(f"Warning: could not copy SE sheet for AM_frac {am_fraction:.2f}, diameter {am_diameter}: {e}")

    print("Simulation complete. All results consolidated into", excel_filename)

