# Code and algorithm design by Paolo Fazzini (paolo.fazzini@cnr.it)

from data_setup import create_sequences, split_data, split_data_multitest
from models import ANNModel, create_model
import numpy as np
from sklearn.metrics import root_mean_squared_error
import tensorflow as tf

from tensorflow.keras.utils import plot_model
#from keras.utils import plot_model
import copy
from tensorflow.keras.callbacks import EarlyStopping, Callback
#from keras.callbacks import EarlyStopping, Callback

class WeightHistory(Callback):
    def __init__(self):
        super().__init__()
        self.weights_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        # Capture the model weights at the end of the epoch
        layer_weights = [layer.get_weights() for layer in self.model.layers]
        self.weights_per_epoch.append(layer_weights)


def _predict(testX, ann_forecasting_steps, time_steps_in, num_in_streams, num_out_streams, model_item):
    # Direct prediction
    len_data = len(testX)
    test_predict_direct = [] # Note: take it from here: this has to become a list of length num_out_streams of lists, same goes for the recursive case
    for i in range(0, len_data, ann_forecasting_steps):
        prediction = model_item.ANN_model.predict(testX[i].reshape(1, time_steps_in, num_in_streams))
        test_predict_direct.extend(prediction)
    test_predict_direct = np.array(test_predict_direct).reshape(len_data, num_out_streams)
    
    # Recursive forecasting
    test_data = np.copy(testX[0])
    predictions = []
    for _ in range(0, len_data, ann_forecasting_steps):
        prediction = model_item.ANN_model.predict(test_data.reshape(1, time_steps_in, num_in_streams))
        predictions.extend(prediction)
        for k in range(num_out_streams):
            test_data[:, k] = np.roll(test_data[:, k], -ann_forecasting_steps, axis=0)
            test_data[-ann_forecasting_steps:,k] = np.vstack(prediction[k])

    test_predict_recursive = np.array(predictions).reshape(len_data, num_out_streams)
    return test_predict_direct, test_predict_recursive







def do_multi_forecasting(
                            id_run,
                            id_stream_to_forecast, 
                            datain_, datain_tr_, 
                            num_test_samples, 
                            num_tests, 
                            time_steps_in, 
                            ann_forecasting_steps, 
                            increment_x, 
                            model_type, 
                            scaler_, 
                            num_in_streams, 
                            num_out_streams, 
                            num_epochs, 
                            do_early_stopping, 
                            ensemble_method, 
                            heuristic_weights = None,
                            heuristic_method = None,
                            init_weight_mode = None
                        ):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    list_models = []
    datain = copy.deepcopy(datain_)    
    datain_tr = copy.deepcopy(datain_tr_)    
    scaler = copy.deepcopy(scaler_)    
                  
    scaler.fit(datain_tr[id_stream_to_forecast].reshape(-1, 1))
    data_scaled = np.concatenate([scaler.transform(array.reshape(-1, 1)) for array in datain], axis=1)
    data_scaled_tr = np.concatenate([scaler.transform(array.reshape(-1, 1)) for array in datain_tr], axis=1)


    # Note: This has been tested only for ann_forecasting_steps=1
    # Note: in case increment_x > 1, adjacent data of data_scaled would be 
    # set apart in dataY of increment_x positions
    dataX, dataY = create_sequences(data_scaled, time_steps_in, ann_forecasting_steps, increment_x)
    trainX, trainY = create_sequences(data_scaled_tr, time_steps_in, ann_forecasting_steps, increment_x)


    _, testsX = split_data_multitest(dataX, num_tests, num_test_samples)
    _, testsY = split_data_multitest(dataY, num_tests, num_test_samples)

    


    if model_type == "ensemble":
        # Create instances of the Model class
        simple_lstm = ANNModel('simple_lstm', 'lstm', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple lstm dir', 'simple lstm rec'], heuristic_weights, heuristic_method, init_weight_mode)
        simple_esn = ANNModel('simple_esn', 'esn', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple esn dir', 'simple esn rec'], heuristic_weights, heuristic_method, init_weight_mode)
        simple_rnn = ANNModel('simple_rnn', 'rnn', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple rnn dir', 'simple rnn rec'], heuristic_weights, heuristic_method, init_weight_mode)
        simple_gru = ANNModel('simple_gru', 'gru', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple gru dir', 'simple gru rec'], heuristic_weights, heuristic_method, init_weight_mode)
        simple_mlp = ANNModel('simple_mlp', 'mlp', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple mlp dir', 'simple mlp rec'], heuristic_weights, heuristic_method, init_weight_mode)



        # Store the models in a list
        list_models = [simple_lstm, simple_esn, simple_rnn, simple_gru, simple_mlp]

    elif model_type == "lstm":
        list_models = [ANNModel('simple_lstm', 'lstm', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple lstm dir', 'simple lstm rec'], heuristic_weights, heuristic_method, init_weight_mode)]
    elif model_type == "mlp":
        list_models = [ANNModel('simple_mlp', 'mlp', 40, num_in_streams, num_out_streams, ann_forecasting_steps, ['tanh', 'tanh'], num_epochs, ['simple mlp dir', 'simple mlp rec'], heuristic_weights, heuristic_method, init_weight_mode)]

    for model_item in list_models:
        np.random.seed(0)
        tf.random.set_seed(0)
        create_model(
            model_item,
            input_shape=(time_steps_in, model_item.hyperparameters['in_vars'])
        )


        model_item.ANN_model.compile(loss=[model_item.loss]*model_item.hyperparameters['out_vars'], optimizer='adam')
        
        
        plot_model(
            model_item.ANN_model, 
            to_file=f'{model_item.name}.png', 
            show_shapes=True, 
            show_layer_names=True
        )
        
        callbacks = []
        
        if do_early_stopping:
            # Ensure EarlyStopping is a proper Callback
            early_stop = EarlyStopping(monitor='loss', patience=1)
            callbacks.append(early_stop)

        if id_run == (0, 0, 0):
            # Ensure WeightHistory is added only when needed
            weight_history_callback = WeightHistory()
            callbacks.append(weight_history_callback)
        else:
            weight_history_callback = None

        if num_out_streams == 1:
            trainY_ = trainY[:,:, id_stream_to_forecast]
        elif num_out_streams == num_in_streams:  
            trainY_ = np.squeeze(trainY)
        else:
            raise Exception("Unmanaged number of output streams")
        
        # Capture the history during training
        model_item.loss_history = model_item.ANN_model.fit(
            trainX, trainY_,
            epochs=model_item.hyperparameters['epochs'],
            batch_size=40,
            verbose=2,
            callbacks=callbacks if callbacks else None
        )
        

        results = []
        for j in range(num_tests):
            test_predict_direct, test_predict_recursive = _predict(testsX[j], 
                                                                ann_forecasting_steps, 
                                                                time_steps_in, 
                                                                num_in_streams, 
                                                                num_out_streams, 
                                                                model_item)


            model_item.predict_direct.append(np.concatenate([scaler.inverse_transform(array.reshape(-1, 1)) for array in test_predict_direct], axis=1).T)
            model_item.predict_recursive.append(np.concatenate([scaler.inverse_transform(array.reshape(-1, 1)) for array in test_predict_recursive], axis=1).T)
    


    for j in range(num_tests):

        testsY[j] = scaler.inverse_transform(testsY[j][:,0,0].reshape(-1,1))



        if model_type == "ensemble":
            # Compute Ensemble
            if ensemble_method == 0: # Trivial: just average the models' forecast
                result = [0] * num_test_samples
                count_contrib = 0
                for model in list_models:    
                    if model.ensemble:
                        count_contrib += 1
                        result = [result[i] + model.predict_recursive[j][i] for i in range(len(result))]

                if count_contrib > 0:
                    result = [result[i]/count_contrib for i in range(len(result))]
            else:
                print("No other ensemble mathods here")
                print("End.")
        else:
            result = model_item.predict_recursive[j]
        
        results.append(result)

    del datain, datain_tr, data_scaled, data_scaled_tr, dataX, dataY, trainX, trainY, scaler   

    return results, testsY, list_models, weight_history_callback

