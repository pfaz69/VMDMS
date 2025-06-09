# Code and algorithm design by Paolo Fazzini (paolo.fazzini@cnr.it)


import os
import numpy as np
import tensorflow as tf

def setup_deterministic_environment(force_cpu):
    
    # Basic environment settings
    os.environ['PYTHONHASHSEED'] = '0'
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
    
    # Force deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_USE_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = 'True'
    
    # Thread control
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # TF specific settings
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)
    
    # Disable mixed precision
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Force synchronous execution
    tf.config.experimental.set_synchronous_execution(True)




from VMD_p import VMD_pers
from sklearn.preprocessing import MinMaxScaler
import argparse
from data_setup import load_data

from forecasting import do_multi_forecasting
from data_setup import create_sequences, split_data_multitest
from plot_results import plot_N_graphs, plot_cumulative_RMSEs, compute_n_plot_spectrum, plot_dots_and_subsets, plot_main_graphs


import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# ***************** FLAGS

just_do_the_RMSE_graph = False

do_early_stopping = True
do_use_granger_weights = False

use_multiprocessing = False
force_CPU = True


setup_deterministic_environment(force_CPU)

def setup_mode_forecasting(
    method,
    i_stream,
    k_mode,
    mode_llst,
    mode_tr_llst,
    heuristic_tr_lllst,
    stream_modes,
    stream_modes_tr,
    do_use_granger_weights=False,
    raw_data=None,
    raw_data_tr=None
):
    """
    Sets up the configuration for N-1 or 1-1 Mode Forecasting based on the selected method.
    
    
    Returns:
        dict: A dictionary containing the forecasting configuration.
    """


    config = {}

    if method == 'NO_DECOMP':
        # Direct forecasting without decomposition
        config['input_modes'] = [raw_data[i_stream]]
        config['input_modes_tr'] = [raw_data_tr[i_stream]]
        config['num_in_streams'] = 1  # Single input (the original signal)
        config['stream_to_forecast'] = 0  # The only one
        config['num_out_streams'] = 1
        config['heuristic_weights'] = None
        config['heuristic_method'] = None
        config['init_weight_mode'] = None
    elif method == 'VMDMS':
        # Perform modified N-1 Mode Forecasting
        config['input_modes'] = mode_llst[i_stream][k_mode][:, :]
        config['input_modes_tr'] = mode_tr_llst[i_stream][k_mode][:, :]
        config['num_in_streams'] = len(mode_tr_llst[i_stream][k_mode])
        config['stream_to_forecast'] = 0  # The first one
        config['num_out_streams'] = 1  # Switch here to multiple outputs
        config['heuristic_weights'] = None
        config['heuristic_method'] = None
        config['init_weight_mode'] = None
    
    elif method == 'VMD':
        # Perform 1-1 Mode Forecasting
        config['input_modes'] = [stream_modes[i_stream, k_mode, :]]
        config['input_modes_tr'] = [stream_modes_tr[i_stream, k_mode, :]]
        config['num_in_streams'] = 1
        config['num_out_streams'] = 1
        config['stream_to_forecast'] = 0  # The only one
        config['heuristic_weights'] = None
        config['heuristic_method'] = None
        config['init_weight_mode'] = None

    return config

def make_n_plot_partitions(base_dir,file_spectrum, stream_modes, stream_modes_tr, stream_freqs_tr, stream_hatmodes_tr, thsh):

    bandwidth_tr_lst = compute_n_plot_spectrum(stream_hatmodes_tr, 
                                            stream_freqs_tr, file_spectrum, do_show = False)
    freqs_tr = [freq_tr for freqs_tr in stream_freqs_tr for freq_tr in freqs_tr] #np.squeeze(omegas.reshape(-1, 1))
    modes_tr = [mode_tr for modes_tr in stream_modes_tr for mode_tr in modes_tr]
    modes = [mode for modes in stream_modes for mode in modes]

    # index 0->streams 1->modes to forecast for each stream 2->stream modes to be used 
    # for forecasting each mode
    mode_llst = [] 
    mode_tr_llst = []
    freq_tr_llst = [] 
    freq_id_subsets_tr_llst = [] 
    for i_stream_tr, mode_freqs_tr in enumerate(stream_freqs_tr):
        freq_tr_lst, mode_tr_lst, mode_lst, freq_id_subsets_tr_lst = do_partition(i_stream_tr, mode_freqs_tr, freqs_tr, modes_tr, modes, thsh, bandwidth_tr_lst)
        mode_tr_llst.append(mode_tr_lst)
        mode_llst.append(mode_lst)
        freq_tr_llst.append(freq_tr_lst)
        freq_id_subsets_tr_llst.append(freq_id_subsets_tr_lst)
        file_subsets = f'{base_dir}/VMD_freq_subsets_{i_stream_tr}.png'
        plot_dots_and_subsets(stream_freqs_tr, freq_tr_lst, thsh/10, file_subsets, do_show = False)

    return mode_tr_llst, mode_llst, freq_tr_llst, freq_id_subsets_tr_llst




def do_partition(i_stream_tr, target_stream_freqs_tr, streams_freqs_tr, stream_modes_tr, stream_modes, thsh, bandwidth_lst):
    freq_subsets_tr = []
    freq_id_subsets_tr = [] 
    mode_subsets_tr = []
    mode_subsets = []

    for i in range(len(target_stream_freqs_tr)):
        # Initialize the subsets with the forecasting target elements: they should be the first element of each list
        # Because of the way the per-set forecasting is organized
        assert(streams_freqs_tr[i_stream_tr*len(target_stream_freqs_tr) + i] == target_stream_freqs_tr[i])
        freq_subset_tr = [streams_freqs_tr[i_stream_tr*len(target_stream_freqs_tr) + i]]
        freq_id_subset_tr = [i_stream_tr*len(target_stream_freqs_tr) + i] # This is only useful for documentation purposes
        mode_subset_tr = [stream_modes_tr[i_stream_tr*len(target_stream_freqs_tr) + i]]
        mode_subset = [stream_modes[i_stream_tr*len(target_stream_freqs_tr) + i]]

        for j in range(len(streams_freqs_tr)):
            if j != i_stream_tr*len(target_stream_freqs_tr) + i: # Don't want to include the same elements twice
                distance = np.linalg.norm(target_stream_freqs_tr[i] - streams_freqs_tr[j])

                if distance < thsh*bandwidth_lst[j]:
                    freq_subset_tr.append(np.array(streams_freqs_tr[j]))
                    freq_id_subset_tr.append(j)
                    mode_subset_tr.append(np.array(stream_modes_tr[j]))
                    mode_subset.append(np.array(stream_modes[j]))

        freq_subsets_tr.append(np.array(freq_subset_tr))
        freq_id_subsets_tr.append(np.array(freq_id_subset_tr))
        mode_subsets_tr.append(np.array(mode_subset_tr))
        mode_subsets.append(np.array(mode_subset))


    return freq_subsets_tr, mode_subsets_tr, mode_subsets, freq_id_subsets_tr


def forecast_task(i_method, method, i_stream, k, mode_llst, mode_tr_llst, stream_modes, stream_modes_tr, do_use_granger_weights, heuristic_tr_lllst, num_test_samples, num_tests, time_steps_in, forecasting_steps, increment_x, model_type, scaler, nums_epochs, do_early_stopping, ensemble_method, raw_data=None, raw_data_tr=None):
    setup_deterministic_environment(force_CPU)
    # Call the function with the required parameters
    forecasting_config = setup_mode_forecasting(
        method=method,
        i_stream=i_stream,
        k_mode=k,
        mode_llst=mode_llst,
        mode_tr_llst=mode_tr_llst,
        heuristic_tr_lllst=heuristic_tr_lllst,
        stream_modes=stream_modes,
        stream_modes_tr=stream_modes_tr,
        do_use_granger_weights=do_use_granger_weights,
        raw_data=raw_data,
        raw_data_tr=raw_data_tr
    )

    # Extract configuration values
    input_modes = forecasting_config['input_modes']
    input_modes_tr = forecasting_config['input_modes_tr']
    num_in_streams = forecasting_config['num_in_streams']
    num_out_streams = forecasting_config['num_out_streams']
    stream_to_forecast = forecasting_config['stream_to_forecast']
    heuristic_weights = forecasting_config['heuristic_weights']
    heuristic_method = forecasting_config['heuristic_method']
    init_weight_mode = forecasting_config['init_weight_mode']

    # For NO_DECOMP, always use the first time_steps_in value
    time_step = time_steps_in[0] if method == 'NO_DECOMP' else time_steps_in[k]

    list_models = []
    mode_result, testY_i, list_models, weight_history_callback = do_multi_forecasting(
                                                            (i_method, i_stream, k),
                                                            stream_to_forecast, 
                                                            input_modes, 
                                                            input_modes_tr, 
                                                            num_test_samples, 
                                                            num_tests, 
                                                            time_step, 
                                                            forecasting_steps, 
                                                            increment_x, 
                                                            model_type, 
                                                            scaler, 
                                                            num_in_streams, 
                                                            num_out_streams, 
                                                            nums_epochs, 
                                                            do_early_stopping, 
                                                            ensemble_method,
                                                            heuristic_weights,
                                                            heuristic_method, 
                                                            init_weight_mode)
    
    return mode_result, testY_i, list_models, weight_history_callback



# SET UP GENERAL VARIABLES & DIRECTORIES ---------------------------------------------------------------------------------

start_time = time.time()

weight_history = None

# PARSING
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--chosen_dataset', type=str, default='Artificial2D_1')
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--init', type=int, default=4)
parser.add_argument('--epochs', type=int, default=15)

# Parse the arguments
args = parser.parse_args()

# CHOOSE DATASET
chosen_dataset = args.chosen_dataset 

if chosen_dataset == "Artificial2D_1":
    time_series_target = ['one', 'two', 'three']#, 'four']
    num_test_samples = 48
else:
    print("No other datasets here")
    print("End.")
    exit()


# SET VARIABLES
ns = len(time_series_target)
num_tests = 3#9#
do_show = True

     
scaler = MinMaxScaler(feature_range=(-.6, .6)) 

# SET ANN HYPERPARAMETERS
model_type = "lstm"
ensemble_method = 0
increment_x = 1  
nums_epochs = args.epochs

time_steps_in = (150, 140, 130, 120, 100, 80, 60, 40, 30, 25, 20, 20) 
ann_forecasting_steps = 1

# VMD PARAMETERS
alpha = 2000       
tau = args.tau 
K = 3

omega0 = None
DC = 0           
init = args.init          
tol = 1e-7   

thsh_partition = .25 

# Set up Forecasting options 
perform_forecasting = ['VMD','VMDMS']
num_methods = len(perform_forecasting)

# ***************** PREPARE GENERAL RESULT FOLDERS
dir_plot_lst = []
Dirdata = f'H{num_test_samples}_tau{tau}_init{init}_ep{nums_epochs}_es{int(do_early_stopping)}_scale{scaler.feature_range}_thshPart{thsh_partition}_GrW{int(do_use_granger_weights)}_K{K}'
for target in time_series_target:
    dirs_plot = []
    for i_method in range(num_methods):
        if perform_forecasting[i_method] is not None:
            dirs_plot += [
                            f'Results/{chosen_dataset}/{Dirdata}/{target}/{perform_forecasting[i_method]}/Graphs', 
                            f'Results/{chosen_dataset}/{Dirdata}/{target}/{perform_forecasting[i_method]}/RMSEs', 
                            f'Results/{chosen_dataset}/{Dirdata}/{target}/{perform_forecasting[i_method]}/Stats',
                        ]
    dirs_plot += [f'Results/{chosen_dataset}/{Dirdata}/{target}/X_Res']
    for item in dirs_plot:
        os.makedirs(item, exist_ok=True)
    dir_plot_lst.append(dirs_plot)

base_plot_dir_elabs = f'Results/{chosen_dataset}/{Dirdata}/Elabs'
os.makedirs(base_plot_dir_elabs, exist_ok=True)



if not just_do_the_RMSE_graph:

    # Initialize result dictionaries
    results_dict = {}
    tests_dict = {}

    # df = load_data_old()
    # data = np.array(df.values.T)

    data, _, _, _ = load_data()


    # The following is needed because VMD wants even sized signals: 
    # in case of "oddity", keep the most recent data
    data = data[:, -(data.shape[1]//2)*2:]




    # ***************** Perform Decomposition

    b_decomposition_performed = False
    stream_modes_dict = {}
    stream_hatmodes_dict = {}
    stream_freqs_dict = {}
    stream_modes_tr_dict = {}
    stream_hatmodes_tr_dict = {}
    stream_freqs_tr_dict = {}
    
    # Create raw data dictionary for NO_DECOMP
    raw_data_dict = {}
    raw_data_tr_dict = {}
    
    for algorithm in perform_forecasting:
        if algorithm == 'NO_DECOMP':
            # Store the raw data for direct forecasting
            raw_data_dict[algorithm] = data
            raw_data_tr_dict[algorithm] = data[:, :-num_tests*num_test_samples]
            
            # Create placeholder dictionaries to maintain consistency with other methods
            # These won't be used for forecasting but help keep the code structure
            stream_modes_dict[algorithm] = np.zeros((len(data), 1, data.shape[1]))  # Just one "mode" which is the signal itself
            stream_hatmodes_dict[algorithm] = np.zeros((len(data), 1, data.shape[1]))  # Placeholder
            stream_freqs_dict[algorithm] = np.zeros((len(data), 1))  # Placeholder
            stream_modes_tr_dict[algorithm] = np.zeros((len(data), 1, data[:, :-num_tests*num_test_samples].shape[1]))
            stream_hatmodes_tr_dict[algorithm] = np.zeros((len(data), 1, data[:, :-num_tests*num_test_samples].shape[1]))
            stream_freqs_tr_dict[algorithm] = np.zeros((len(data), 1))  # Placeholder
            
            print(f"Done storing raw data for NO_DECOMP")
        elif algorithm in ('VMD', 'VMDMS'):
            if b_decomposition_performed == False:
                b_decomposition_performed = True

                stream_modes = []
                stream_hatmodes = []
                stream_freqs = []
                stream_modes_tr = []
                stream_hatmodes_tr = []
                stream_freqs_tr = []
                for i, item in enumerate(data):
                    u, u_hat, omega = VMD_pers(item, alpha, tau, K, DC, init, tol, omega0)
                    u_tr, u_hat_tr, omega_tr = VMD_pers(item[:-num_tests*num_test_samples], 
                                                        alpha, tau, K, DC, init, tol, omega0)

                    stream_modes.append(u)
                    stream_hatmodes.append(u_hat.T) 
                    stream_freqs.append(omega[-1])
                    stream_modes_tr.append(u_tr)
                    stream_hatmodes_tr.append(u_hat_tr.T)
                    stream_freqs_tr.append(omega_tr[-1])
                    print(f"Done VMD of item {i+1} of {len(data)}")

                # ...Now convert lists to arrays
                stream_modes_ = np.array(stream_modes)
                stream_hatmodes_ = np.array(stream_hatmodes)
                stream_freqs_ = np.array(stream_freqs)
                stream_modes_tr_ = np.array(stream_modes_tr)
                stream_hatmodes_tr_ = np.array(stream_hatmodes_tr)
                stream_freqs_tr_ = np.array(stream_freqs_tr)

                # Stick into the dict
                stream_modes_dict[algorithm] = stream_modes_    
                stream_hatmodes_dict[algorithm] = stream_hatmodes_  
                stream_freqs_dict[algorithm] = stream_freqs_  
                stream_modes_tr_dict[algorithm] = stream_modes_tr_  
                stream_hatmodes_tr_dict[algorithm] = stream_hatmodes_tr_  
                stream_freqs_tr_dict[algorithm] = stream_freqs_tr_
            else: 
                for processed_alg in ('VMD', 'VMDMS'):
                    if stream_modes_dict.get(processed_alg) is not None:
                        # Found a previously processed algorithm, use its results
                        stream_modes_dict[algorithm] = stream_modes_dict[processed_alg]
                        stream_hatmodes_dict[algorithm] = stream_hatmodes_dict[processed_alg]
                        stream_freqs_dict[algorithm] = stream_freqs_dict[processed_alg]
                        stream_modes_tr_dict[algorithm] = stream_modes_tr_dict[processed_alg]
                        stream_hatmodes_tr_dict[algorithm] = stream_hatmodes_tr_dict[processed_alg]
                        stream_freqs_tr_dict[algorithm] = stream_freqs_tr_dict[processed_alg]
                        break  
            
            
            # Convert lists to arrays (following the pattern used in VMD section)
            stream_modes_dict[algorithm] = np.array(stream_modes)
            stream_hatmodes_dict[algorithm] = np.array(stream_hatmodes)
            stream_freqs_dict[algorithm] = np.array(stream_freqs)
            stream_modes_tr_dict[algorithm] = np.array(stream_modes_tr)
            stream_hatmodes_tr_dict[algorithm] = np.array(stream_hatmodes_tr)
            stream_freqs_tr_dict[algorithm] = np.array(stream_freqs_tr)
            
    

    
    # Initialize result dictionaries - each method will have a dict entry
    results_dict = {}
    tests_dict = {}
    
    id_dir = 0
    num_type_graphs = 3
    for i_method in range(num_methods):
        method_name = perform_forecasting[i_method]
        results_dict[method_name] = {}
        
        if method_name == 'VMDMS':
            # ***************** MAKE AND PLOT PARTITIONS

            file_spectrum = f'{base_plot_dir_elabs}/VMD_spectrum_{method_name}.png'
            mode_tr_llst, mode_llst, freq_tr_llst, freq_id_subsets_tr_llst = make_n_plot_partitions(base_plot_dir_elabs, file_spectrum, stream_modes_dict[method_name], stream_modes_tr_dict[method_name], stream_freqs_tr_dict[method_name], stream_hatmodes_tr_dict[method_name], thsh_partition)

        else:
            heuristic_tr_lllst = None
            mode_tr_llst = None
            mode_llst = None
            freq_tr_llst = None
            freq_id_subsets_tr_llst = None
        
        for i_stream in range(len(data)):
            stream_name = time_series_target[i_stream]
            results_dict[method_name][stream_name] = {}
            
            # For NO_DECOMP, only need to forecast once, not K times
            if method_name == 'NO_DECOMP':
                k_range = [0]  # Just one forecast
            else:
                k_range = range(K)  # K forecasts for each mode
                
            mode_results = []
            testY_ = []
            
            if method_name != 'NAIVE_PERSISTENCE' and method_name != 'NAIVE_SEASONAL':
                if use_multiprocessing:
                    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        # Create a list to store futures
                        futures = []
                        
                        # Submit tasks with explicit task_id
                        for k in k_range:
                            future = executor.submit(
                                forecast_task, 
                                i_method, method_name, i_stream, k, mode_llst, mode_tr_llst, 
                                stream_modes_dict[method_name], stream_modes_tr_dict[method_name], do_use_granger_weights, 
                                heuristic_tr_lllst, num_test_samples, num_tests, 
                                time_steps_in, ann_forecasting_steps, increment_x, 
                                model_type, scaler, nums_epochs, do_early_stopping, 
                                ensemble_method, raw_data_dict.get(method_name), raw_data_tr_dict.get(method_name)
                            )
                            futures.append(future)
                        
                        # Process results in order of completion                        
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                mode_result, testY_i, list_models, weight_history_callback = future.result()
                                mode_results.append(mode_result)
                                testY_.append(testY_i)
                                if weight_history_callback is not None:
                                    weight_history = weight_history_callback
                                del mode_result, testY_i
                            except Exception as e:
                                print(f"Task failed with error: {e}")


                    for dir_plot in dir_plot_lst[i_stream]:
                        if not os.path.exists(dir_plot):
                            os.makedirs(dir_plot)

                    for k in range(len(k_range)):
                        plot_main_graphs(
                            k,
                            num_tests=num_tests,
                            models=list_models,
                            target_graph=testY_[k],
                            ensemble=mode_results[k],
                            label_ensemble=model_type,
                            dir_base_1=dir_plot_lst[i_stream][id_dir],
                            dir_base_2=dir_plot_lst[i_stream][id_dir + 1],
                            dir_base_3=dir_plot_lst[i_stream][id_dir + 2]
                        )

                else:
                    for k in k_range:
                        # Call the function with the required parameters
                        forecasting_config = setup_mode_forecasting(
                            method=method_name,
                            i_stream=i_stream,
                            k_mode=k,
                            mode_llst=mode_llst,
                            mode_tr_llst=mode_tr_llst,
                            heuristic_tr_lllst=heuristic_tr_lllst,
                            stream_modes=stream_modes_dict[method_name],
                            stream_modes_tr=stream_modes_tr_dict[method_name],
                            do_use_granger_weights=do_use_granger_weights,
                            raw_data=raw_data_dict.get(method_name),
                            raw_data_tr=raw_data_tr_dict.get(method_name)
                        )

                        # Extract configuration values
                        input_modes = forecasting_config['input_modes']
                        input_modes_tr = forecasting_config['input_modes_tr']
                        num_in_streams = forecasting_config['num_in_streams']
                        num_out_streams = forecasting_config['num_out_streams']
                        stream_to_forecast = forecasting_config['stream_to_forecast']
                        heuristic_weights = forecasting_config['heuristic_weights']
                        heuristic_method = forecasting_config['heuristic_method']
                        init_weight_mode = forecasting_config['init_weight_mode']

                        # For NO_DECOMP, always use the first time_steps_in value
                        time_step = time_steps_in[0] if method_name == 'NO_DECOMP' else time_steps_in[k]

                        list_models = []
                        mode_result, testY_i, list_models, weight_history_callback = do_multi_forecasting(
                                                                                    (i_method, i_stream, k),
                                                                                    stream_to_forecast, 
                                                                                    input_modes, 
                                                                                    input_modes_tr, 
                                                                                    num_test_samples, 
                                                                                    num_tests, 
                                                                                    time_step, 
                                                                                    ann_forecasting_steps, 
                                                                                    increment_x, 
                                                                                    model_type, 
                                                                                    scaler, 
                                                                                    num_in_streams, 
                                                                                    num_out_streams, 
                                                                                    nums_epochs, 
                                                                                    do_early_stopping, 
                                                                                    ensemble_method,
                                                                                    heuristic_weights,
                                                                                    heuristic_method, 
                                                                                    init_weight_mode)
                        if weight_history_callback is not None:
                            weight_history = weight_history_callback
                        mode_results.append(mode_result) 
                        testY_.append(testY_i)
                        
                        for dir_plot in dir_plot_lst[i_stream]:
                            if not os.path.exists(dir_plot):
                                os.makedirs(dir_plot)

                        plot_main_graphs(
                            k,
                            num_tests=num_tests,
                            models=list_models,
                            target_graph=testY_i,
                            ensemble=mode_result,
                            label_ensemble=model_type,
                            dir_base_1=dir_plot_lst[i_stream][id_dir],
                            dir_base_2=dir_plot_lst[i_stream][id_dir + 1],
                            dir_base_3=dir_plot_lst[i_stream][id_dir + 2]
                            )
        
            # Sum up the forecasts to get the forecast for the original signal
            # For NO_DECOMP, there's only one result, so no need to sum
            if method_name == 'NO_DECOMP':
                # For NO_DECOMP, no need to sum as there's only one mode
                forecast_signals = np.array([modes[:,0] for modes in mode_results[0]])
            elif method_name == "NAIVE_PERSISTENCE":
                forecast_signals = [data[i_stream][-(num_tests - i_test + 1) * num_test_samples:-(num_tests - i_test) * num_test_samples] for i_test in range(num_tests)]
            elif method_name == "NAIVE_SEASONAL":
                seasonality = 3 # Corresponding to 24h
                forecast_signals = [data[i_stream][-(num_tests - i_test + seasonality) * num_test_samples:-(num_tests - i_test + seasonality - 1) * num_test_samples] for i_test in range(num_tests)]
            else:
                # For decomposition methods, sum all modes
                mode_results = [[modes[:,0] for modes in tests] for tests in mode_results]
                forecast_signals = np.sum(np.array(mode_results), axis=0)
                
            # lot Comparative Results: needs to create test data for the whole signal
            # In the following, any number of steps will do as we are only interested
            # in the test set. Chosen 0.
            _, dataY = create_sequences(np.expand_dims(data[i_stream], 1), time_steps_in[0], ann_forecasting_steps, increment_x)
            _, testsY = split_data_multitest(dataY, num_tests, num_test_samples)

            # Store test data only once per stream
            if stream_name not in tests_dict:
                tests_dict[stream_name] = {}

            for j in range(num_tests):
                test_id = f"test_{j}"
                
                # Store the forecast and test results in dictionaries
                results_dict[method_name][stream_name][test_id] = forecast_signals[j]
                
                # Store test data only once per test
                if test_id not in tests_dict[stream_name]:
                    tests_dict[stream_name][test_id] = testsY[j][:,0]
                
                graphs = [forecast_signals[j], testsY[j][:,0]]
                labels = [method_name, 'Target']
                plot_N_graphs(j, graphs, labels, dir_plot_lst[i_stream][id_dir], model_type, 'test')

        # Increase the dir id to pick the right directories            
        id_dir += num_type_graphs
    
    np.savez(f'Results/{chosen_dataset}/{Dirdata}/arrays_dict.npz', 
             results_dict=results_dict, 
             tests_dict=tests_dict)
    

data_load = np.load(f'Results/{chosen_dataset}/{Dirdata}/arrays_dict.npz', allow_pickle=True)
results_dict = data_load['results_dict'].item()
tests_dict = data_load['tests_dict'].item()
print("Loaded results from dictionary format")
    

labels = list(perform_forecasting)

# Initialize an empty list to store all error values
all_rmse_values = []
all_mae_values = []
all_smape_values = []

for i_stream, target in enumerate(time_series_target):
    for j in range(num_tests):
        test_id = f"test_{j}"
        
        # Plot Signals' Graphs
        signals = []
        for method_name in perform_forecasting:
            # Get forecast result from dictionary
            forecast = results_dict[method_name][target][test_id]
            signals.append(np.squeeze(forecast).reshape(-1,1))
        
        # Get ground truth from tests dictionary
        observation = [np.squeeze(tests_dict[target][test_id]).reshape(-1,1)]


        all_signals = signals + observation
        all_labels = labels+["Target"]
        plot_N_graphs(
                        0, 
                        all_signals, 
                        all_labels, 
                        f'Results/{chosen_dataset}/{Dirdata}/{target}/X_Res',
                        f'graph_stream_{i_stream}_{j}',  
                        type_graph = 'test'
                        )
        
        # Plot Signals' RMSE Histograms
        file_out = f"Results/{chosen_dataset}/{Dirdata}/{target}/X_Res/RMSE_stream_{i_stream}_test_{j}"
        rmse_values, mae_values, smape_values, _, _ = plot_cumulative_RMSEs(
                                    signals, 
                                    observation,
                                    labels,
                                    file_out,
                                    )
        
        all_rmse_values.append(rmse_values)
        all_mae_values.append(mae_values)
        all_smape_values.append(smape_values)



print("--- %s minutes ---" % ((time.time() - start_time) / 60))