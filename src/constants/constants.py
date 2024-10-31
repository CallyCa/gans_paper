# src/constants.py
import tensorflow as tf

# Caminhos de arquivos e diretórios
DEFAULT_BASE_CONFIG_PATH = './config/config.yaml'
DEFAULT_RESULTS_DIR = './results/gan_experiments'
MEMORY_USAGE_FILE = 'memory_usage.csv'
MEMORY_USAGE_PLOT_FILE = 'memory_usage.png'
METRICS_CSV_FILE = 'metrics.csv'
CONFIG_JSON_FILE = 'config.json'
CONFIG_YAML_FILE = 'config.yml'
SUMMARY_RESULTS_FILE = 'summary_results.csv'
EXPERIMENT_IMAGES_TEMPLATE = "experiment_{experiment}_epoch_{epoch}_generated_images.png"

# Parâmetros de treinamento
DEFAULT_N_EXPERIMENTS = 2
DEFAULT_CHECKPOINT_INTERVAL = 5

# Mensagens de log e saída
START_EXPERIMENT_MESSAGE = "\nIniciando Experimento {current}/{total}"
EXPERIMENT_COMPLETED_MESSAGE = "Experimento {experiment} concluído. Métricas e configuração salvas em {directory}."
NO_VARIABILITY_WARNING = "Aviso: Experimento {experiment} ignorado devido à baixa variabilidade nas métricas."
EXPERIMENTS_FINISHED_MESSAGE = "\nExperimentos concluídos. Resultados salvos no diretório especificado."
INSUFFICIENT_VARIABILITY_MESSAGE = "Nenhum experimento gerou métricas variáveis o suficiente para serem salvas e analisadas."

# Parâmetros padrão e limites
DEFAULT_DIVERSITY_THRESHOLD = 0.3
MIN_VARIABILITY_THRESHOLD = 0.01
NORMALIZATION_MIN = 0.0
NORMALIZATION_MAX = 1.0
DEFAULT_POOL_PROCESSES = 4  # Ajustável para otimização de CPU

# Mensagens de log
NO_VARIABILITY_INSUFICIENT_WARNING = "Aviso: Variabilidade insuficiente nas métricas calculadas. Métricas não serão salvas."
INSUFFICIENT_VARIABILITY_ERROR = "As métricas calculadas não são variadas o suficiente para calcular a preferência global."
NO_METRICS_TO_SAVE_WARNING = "Nenhuma métrica para salvar."
SAVED_METRICS_MESSAGE = "Métricas salvas em {csv_file_path}"
METRICS_CALCULATED_MESSAGE = "Métricas calculadas:\n{metrics_description}"
AHP_RANKING_MESSAGE = "\nRanking de preferência global com AHP Gaussian:\n{global_preference}"
CHECKPOINT_SAVE_MESSAGE = "Checkpoint salvo com métricas no intervalo definido."
DIVERSITY_THRESHOLD_ADJUST_MESSAGE = "Limiar de diversidade ajustado para {diversity_threshold}"
LPIPS_SCORE_CALCULATION_MESSAGE = "Calculando LPIPS para medir diversidade de imagem."

# Caminho e configurações padrão para checkpoints
DEFAULT_CHECKPOINT_FILENAME = 'gan_checkpoint_epoch-{epoch:02d}.h5'
DEFAULT_MONITOR_METRIC = 'loss'
DEFAULT_MODE = 'min'
DEFAULT_VERBOSITY = 1

# Configurações do GAN e rede neural
IMAGE_SHAPE = (28, 28)  # Dimensões das imagens geradas
IMAGE_CHANNELS = 1      # Número de canais da imagem
GENERATOR_OUTPUT_ACTIVATION = "sigmoid"  # Função de ativação de saída do gerador

# Configuração de otimizadores disponíveis
OPTIMIZERS = {
    "adam": tf.keras.optimizers.Adam,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "sgd": tf.keras.optimizers.SGD
}