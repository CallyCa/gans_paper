import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_runner import ExperimentRunner
from analysis.results_analysis import ResultsAnalyzer
from constants.constants import (
    DEFAULT_BASE_CONFIG_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_N_EXPERIMENTS,
    MEMORY_USAGE_FILE,
    MEMORY_USAGE_PLOT_FILE
)

def main():
    parser = argparse.ArgumentParser(description="Executa experimentos de treinamento GAN ou análise de resultados.")
    parser.add_argument('--mode', choices=['experiment', 'analysis'], default='experiment',
                        help="Modo de execução: 'experiment' para rodar experimentos ou 'analysis' para gerar plots.")
    parser.add_argument('--n_experiments', type=int, default=DEFAULT_N_EXPERIMENTS, help="Número de experimentos a serem executados (aplicável no modo 'experiment').")
    parser.add_argument('--base_config_path', type=str, default=DEFAULT_BASE_CONFIG_PATH, help="Caminho para o arquivo de configuração base.")
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR, help="Diretório para salvar ou carregar resultados.")
    parser.add_argument('--experiment_number', type=int, help="Número do experimento para visualização de imagens geradas.")
    parser.add_argument('--epoch', type=int, help="Época do checkpoint para visualizar as imagens geradas.")

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == 'experiment':
        experiment_runner = ExperimentRunner(args.base_config_path, args.results_dir)
        experiment_runner.run_experiments(args.n_experiments)
    elif args.mode == 'analysis':
        analyzer = ResultsAnalyzer(args.results_dir)
        
        print("\nGerando gráficos de análise de resultados:\n")
        analyzer.plot_metric_distribution('image_quality_ssim')
        analyzer.plot_hyperparam_vs_metric('learning_rate', 'image_quality_fid')
        analyzer.plot_all_metrics()
        analyzer.plot_ranking_vs_score()
        analyzer.plot_loss_convergence()
        analyzer.plot_metrics_heatmap()
        analyzer.plot_metric_trends()
        analyzer.plot_hyperparameter_distributions()
        
        memory_usage_list = pd.read_csv(os.path.join(args.results_dir, MEMORY_USAGE_FILE))['Memory Usage (GB)'].tolist()
        epochs = list(range(1, len(memory_usage_list) + 1))

        analyzer.plot_memory_usage(
            memory_usage_list=memory_usage_list,
            epochs=epochs,
            save_path=os.path.join(args.results_dir, MEMORY_USAGE_PLOT_FILE)
        )
        
        if args.experiment_number and args.epoch:
            analyzer.plot_generated_images(args.experiment_number, args.epoch)

if __name__ == "__main__":
    main()
