from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Dense


def make_model(args):
    model = Sequential()
    model.add(LSTM(args.n_hidden, input_shape=(args.time_steps, args.n_features), return_sequences=True))
    model.add(LSTM(args.n_hidden, return_sequences=False))
    model.add(Dense(args.n_output))
    return model
