"""
"""

#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed
from tensorflow.keras.models import Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


class LSTMAutoencoder:
    def __init__(self):
        self.loss: str = 'mean_squared_error'
        self.optimizer: str = 'adam'
        self.latent_dim: int = 32

        self.model: Model = None

    def create_model(self):
        length_sequence = 1029
        # Define the input shape
        input_shape = (length_sequence, 1)

        # Encoder
        self.encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
        encoder_lstm = LSTM(units=self.latent_dim, return_state=True,
                            name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, 1), name='decoder_inputs')
        decoder_lstm = LSTM(units=self.latent_dim, return_sequences=True,
                            return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=self.encoder_states)

        # Dense layer to produce the output sequence
        decoder_dense = TimeDistributed(
            Dense(1, activation='linear'), name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Model
        self.model = Model(inputs=[self.encoder_inputs, decoder_inputs],
                           outputs=decoder_outputs)

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train(
            self, X_train: np.ndarray, X_val: np.ndarray,
            batch_size: int = 32, save_model: bool = False, epochs: int = 3):
        if save_model:
            checkpoint = ModelCheckpoint(
                'model_weights.h5', save_best_only=True)
            callbacks_list = [checkpoint]

        else:
            callbacks_list = None

        self.create_model()

        self.model.fit(
            [X_train, X_train], X_train,
            validation_data=([X_val, X_val], X_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    def encode(self, X_test) -> np.ndarray:
        self.create_encoder()
        # TODO: check if need to return state_h or state_c
        return self.encoder.predict(X_test)[0]

    def create_encoder(self):
        self.encoder = Model(inputs=self.encoder_inputs,
                             outputs=self.encoder_states)
