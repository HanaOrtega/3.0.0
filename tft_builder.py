# tft_builder.py
# -*- coding: utf-8 -*-
"""
Moduł do budowania modelu Temporal Fusion Transformer (TFT).
Wersja: v1.1 - Poprawiona i Zrefaktoryzowana

Implementacja TFT w Keras, inspirowana oryginalną pracą i przykładami Google.
Poprawki:
- Prawidłowa inicjalizacja warstw w metodzie __init__.
- Poprawny import brakujących warstw (Flatten).
- Prawidłowe przekazywanie stanu między enkoderem a dekoderem LSTM.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Input, Concatenate, Flatten, LSTM
)
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
import logging
from typing import Dict, List, Optional, Tuple
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from config_imports import TimeDistributed

logger = logging.getLogger(__name__)

@register_keras_serializable() # <-- ADD THIS DECORATOR
class GatedResidualNetwork(Layer):
    """Gated Residual Network (GRN) - kluczowy blok TFT."""

    def __init__(self, units: int, dropout_rate: float, name: str = "grn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1 = Dense(self.units, activation='elu')
        self.dense2 = Dense(self.units)
        self.dropout = Dropout(self.dropout_rate)
        self.layer_norm = LayerNormalization()
        self.gate_dense = Dense(self.units, activation='sigmoid')
        self.add = Add()
        self.projection_dense = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.projection_dense = Dense(self.units)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.projection_dense:
            residual = self.projection_dense(inputs)
        else:
            residual = inputs

        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x)

        gating_signal = self.gate_dense(x)
        gated_output = x * gating_signal

        return self.add([residual, gated_output])

@register_keras_serializable()
class VariableSelectionNetwork(Layer):
    """Variable Selection Network (VSN) for feature weighting."""

    def __init__(self, num_features: int, units: int, dropout_rate: float, name: str = "vsn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate
        # This GRN is for calculating weights
        self.grn_for_weights = GatedResidualNetwork(self.units, self.dropout_rate, name="grn_for_weights")
        self.softmax_dense = Dense(self.num_features, activation='softmax')
        self.flatten = Flatten()

        # --- START OF FIX ---
        # This new GRN will process the features at each timestep
        self.grn_for_processing = GatedResidualNetwork(self.units, self.dropout_rate, name="grn_for_processing")
        self.time_distributed_grn = TimeDistributed(self.grn_for_processing)
        # --- END OF FIX ---

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # 'inputs' has shape (batch, seq_len, num_features)
        flattened_inputs = self.flatten(inputs)

        # Calculate weights (this part was correct)
        weights = self.grn_for_weights(flattened_inputs)
        weights = self.softmax_dense(weights)
        weights = tf.expand_dims(weights, axis=1)  # shape -> (batch, 1, num_features)

        # Apply weights to the original 3D input
        weighted_inputs = inputs * weights

        # --- START OF FIX ---
        # Replace the incorrect reduce_sum with the TimeDistributed GRN.
        # This applies the GRN to each timestep, preserving the 3D structure.
        # The output shape will be (batch, seq_len, self.units), which is 3D.
        processed_inputs = self.time_distributed_grn(weighted_inputs)
        # --- END OF FIX ---

        return processed_inputs, weights


def build_tft_model(params: Dict) -> Optional[Model]:
    """Buduje i kompiluje kompletny model Temporal Fusion Transformer."""
    logger.info("--- Budowanie modelu Temporal Fusion Transformer (TFT) ---")
    try:
        seq_len = params['data']['sekwencja_dlugosc']
        horizon = params['data']['horyzont_predykcji']
        d_model = params['tft_d_model']
        num_heads = params['tft_num_heads']
        dropout_rate = params['tft_dropout_rate']

        input_shapes = {
            'observed_past': (seq_len, params['n_features_observed']),
            'known_future': (seq_len + horizon, params['n_features_known'])
        }
        inputs = {name: Input(shape=shape, name=name) for name, shape in input_shapes.items()}

        observed_vsn = VariableSelectionNetwork(params['n_features_observed'], d_model, dropout_rate,
                                                name="observed_vsn")
        observed_processed, _ = observed_vsn(inputs['observed_past'])

        known_vsn = VariableSelectionNetwork(params['n_features_known'], d_model, dropout_rate, name="known_vsn")
        known_processed, _ = known_vsn(inputs['known_future'])

        encoder_input = Concatenate(axis=-1)([observed_processed, known_processed[:, :seq_len]])

        encoder_lstm = LSTM(d_model, return_sequences=True, return_state=True, name="encoder_lstm")
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_input)
        encoder_outputs = Dropout(dropout_rate)(encoder_outputs)  # <-- DODAJ
        decoder_lstm = LSTM(d_model, return_sequences=True, name="decoder_lstm")
        decoder_outputs = decoder_lstm(known_processed, initial_state=[state_h, state_c])
        decoder_outputs = Dropout(dropout_rate)(decoder_outputs)  # <-- DODAJ
        gated_skip = GatedResidualNetwork(d_model, dropout_rate, name="gated_skip_connection")
        gated_skip_output = gated_skip(decoder_outputs)

        attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate, name="mha_attention")
        attention_output = attention(query=gated_skip_output, key=encoder_outputs, value=encoder_outputs)
        attention_output = Dropout(dropout_rate)(attention_output)  # <-- DODAJ
        attention_with_skip = Add()([gated_skip_output, attention_output])
        final_grn_output = GatedResidualNetwork(d_model, dropout_rate, name="final_grn")(attention_with_skip)
        final_grn_output = Dropout(dropout_rate)(final_grn_output)  # <-- DODAJ
        output_sliced = final_grn_output[:, -horizon:, :]
        output_flat = Flatten()(output_sliced)
        output_layer = Dense(horizon, name='wyjscie_predykcji')(output_flat)

        model = Model(inputs=inputs, outputs=output_layer, name="TFT_Model")

        lr = float(params.get('learning_rate', 0.001))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        logger.info("Zakończono budowę i kompilację modelu TFT.")
        model.summary(print_fn=lambda line: logger.info(f"    {line}"))
        return model

    except Exception as e:
        logger.critical(f"KRYTYCZNY BŁĄD podczas budowy modelu TFT: {e}", exc_info=True)
        return None