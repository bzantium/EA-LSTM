from utils import load_data, data_to_series_features, get_data_loader, is_minimum, make_cuda
from evolutionary import (initialize_weights, individual_to_key,
                          pop_to_weights, select, reconstruct_population)
from train import train, evaluate
from model import weightedLSTM
import argparse
from copy import deepcopy
from sklearn.model_selection import train_test_split


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--iterations', type=int, default=20,
                        help="Specify the number of evolution iterations")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Specify batch size")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Specify the number of epochs for adaptation")
    parser.add_argument('--log_step', type=int, default=50,
                        help="Specify log step size for adaptation")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--data', type=str, default='pollution.csv',
                        help="Path to the dataset")
    parser.add_argument('--pop_size', type=int, default=36)
    parser.add_argument('--code_length', type=int, default=6)
    parser.add_argument('--n_select', type=int, default=6)
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--n_features', type=int, default=10)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_arguments()
    data = load_data(args.data)

    features = data_to_series_features(data, args.time_steps)
    train_features, features = train_test_split(features, test_size=0.3)
    valid_features, test_features = train_test_split(features, test_size=0.5)
    train_data_loader = get_data_loader(train_features, args.batch_size)
    valid_data_loader = get_data_loader(valid_features, args.batch_size)
    test_data_loader = get_data_loader(test_features, args.batch_size)

    pop, weights = initialize_weights(args.pop_size, args.time_steps, args.code_length)
    key_to_rmse = {}
    best_model = None
    for iteration in range(args.iterations):
        for enum, (indiv, weight) in enumerate(zip(pop, weights)):
            print('iteration: [%d/%d] indiv_no: [%d/%d]' % (iteration, args.iterations, enum, args.pop_size))
            key = individual_to_key(indiv)
            if key not in key_to_rmse.keys():
                model = weightedLSTM(args.n_features, args.n_hidden, args.n_layers,
                                     args.n_output, weight, args.bidirectional)
                model = make_cuda(model)
                if best_model is not None:
                    model.load_state_dict(best_model.state_dict())
                model = train(args, model, train_data_loader)
                rmse, mae = evaluate(args, model, valid_data_loader)
                if is_minimum(rmse, key_to_rmse):
                    best_model = deepcopy(model)
                key_to_rmse[key] = rmse

        pop_selected, fitness_selected = select(pop, args.n_select, key_to_rmse)
        pop = reconstruct_population(pop_selected, args.pop_size)
        weights = pop_to_weights(pop, args.time_steps, args.code_length)

    print('test evaluation:')
    evaluate(args, best_model, test_data_loader)


if __name__ == '__main__':
    main()
