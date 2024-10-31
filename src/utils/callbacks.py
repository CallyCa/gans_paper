# src/utils/callbacks.py

import os
import tensorflow as tf
from constants.constants import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MONITOR_METRIC,
    DEFAULT_MODE,
    DEFAULT_VERBOSITY
)

def create_checkpoint_callback(checkpoint_dir, monitor=DEFAULT_MONITOR_METRIC, mode=DEFAULT_MODE, verbose=DEFAULT_VERBOSITY):
    """
    Cria e retorna um callback para salvar checkpoints durante o treinamento do modelo.

    Args:
        checkpoint_dir (str): Diretório onde os checkpoints serão salvos.
        monitor (str): Métrica a ser monitorada para salvar o melhor modelo.
        mode (str): Modo de monitoramento ('min' ou 'max').
        verbose (int): Nível de verbosidade.

    Returns:
        tf.keras.callbacks.ModelCheckpoint: Callback configurado para checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, DEFAULT_CHECKPOINT_FILENAME)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,  # Define como False para salvar a cada época
        monitor=monitor,
        mode=mode,
        verbose=verbose
    )
    return checkpoint_callback
