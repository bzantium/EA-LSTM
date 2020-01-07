from utils import (load_data, data_to_series_features,
                   apply_weight, is_minimum)
from algorithm import (initialize_weights, individual_to_key,
                       pop_to_weights, select, reconstruct_population)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model
import argparse
import math
import numpy as np
from model import make_model
from copy import copy
from sklearn.model_selection import train_test_split


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--iterations', type=int, default=20,
                        help="Specify the number of evolution iterations")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Specify batch size")
    parser.add_argument('--initial_epochs', type=int, default=100,
                        help="Specify the number of epochs for initial training")
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="Specify the number of epochs for competitive search")
    parser.add_argument('--log_step', type=int, default=100,
                        help="Specify log step size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--data', type=str, default='pollution.csv',
                        help="Path to the dataset")
    parser.add_argument('--pop_size', type=int, default=36)
    parser.add_argument('--code_length', type=int, default=6)
    parser.add_argument('--n_select', type=int, default=6)
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_arguments()
    data, y_scaler = load_data(args.data)
    args.n_features = np.size(data, axis=-1)
    X, y = data_to_series_features(data, args.time_steps)
    train_X, X, train_y, y = train_test_split(X, y, test_size=0.3)
    valid_X, test_X, valid_y, test_y = train_test_split(X, y, test_size=0.5)

    optimizer = optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.max_grad_norm)
    best_model = make_model(args)
    best_weight = [1.0] * args.time_steps
    best_model.compile(loss='mse', optimizer=optimizer)
    print("Initial training before competitive random search")
    best_model.fit(apply_weight(train_X, best_weight), train_y, epochs=args.initial_epochs,
                   validation_data=(apply_weight(valid_X, best_weight), valid_y), shuffle=True)
    print("\nInitial training is done. Start competitive random search.\n")

    pop, weights = initialize_weights(args.pop_size, args.time_steps, args.code_length)
    key_to_rmse = {}
    for iteration in range(args.iterations):
        for enum, (indiv, weight) in enumerate(zip(pop, weights)):
            print('iteration: [%d/%d] indiv_no: [%d/%d]' % (iteration + 1, args.iterations, enum + 1, args.pop_size))
            key = individual_to_key(indiv)
            if key not in key_to_rmse.keys():
                model = make_model(args)
                model.compile(loss='mse', optimizer=optimizer)
                model.set_weights(best_model.get_weights())
                model.fit(apply_weight(train_X, weight), train_y, epochs=args.num_epochs,
                          validation_data=(apply_weight(valid_X, weight), valid_y), shuffle=True)
                pred_y = model.predict(apply_weight(valid_X, weight))
                inv_pred_y = y_scaler.inverse_transform(pred_y)
                inv_valid_y = y_scaler.inverse_transform(np.expand_dims(valid_y, axis=1))
                rmse = math.sqrt(mean_squared_error(inv_valid_y, inv_pred_y))
                mae = mean_absolute_error(inv_valid_y, inv_pred_y)
                print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))
                if is_minimum(rmse, key_to_rmse):
                    best_model.set_weights(model.get_weights())
                    best_weight = copy(weight)
                key_to_rmse[key] = rmse

        pop_selected, fitness_selected = select(pop, args.n_select, key_to_rmse)
        pop = reconstruct_population(pop_selected, args.pop_size)
        weights = pop_to_weights(pop, args.time_steps, args.code_length)

    print('test evaluation:')
    pred_y = best_model.predict(apply_weight(test_X, best_weight))
    inv_pred_y = y_scaler.inverse_transform(pred_y)
    inv_test_y = y_scaler.inverse_transform(np.expand_dims(test_y, axis=1))
    rmse = math.sqrt(mean_squared_error(inv_test_y, inv_pred_y))
    mae = mean_absolute_error(inv_test_y, inv_pred_y)
    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))


if __name__ == '__main__':
    main()
