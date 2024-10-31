import tensorflow as tf
import os

def check_and_set_device():
    # Verifica se o ROCm está instalado e configurado para GPUs AMD
    if "ROCM_PATH" in os.environ:
        print("ROCm detectado no ambiente. Tentando configurar para execução em GPU AMD...")
    else:
        print("Nenhuma GPU ROCm detectada. Tentando executar em CPU...")

    # Verifica e exibe dispositivos disponíveis
    physical_devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("\nGPU disponível para o TensorFlow.")
        for gpu in gpus:
            print(f"Dispositivo GPU: {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\nNenhuma GPU disponível. TensorFlow executará na CPU.")
    
    # Mostra o dispositivo de execução para operações
    with tf.device('/device:GPU:0' if gpus else '/device:CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("\nResultado de uma operação de teste:")
        print(c)

