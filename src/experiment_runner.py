import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.config_handler import ConfigHandler
from utils.data_loader import DataLoader
from models.gan import GAN
from evaluation.evaluator import Evaluator
from device.check_device import check_and_set_device
from constants.constants import (
    MEMORY_USAGE_FILE,
    EXPERIMENTS_FINISHED_MESSAGE,
    METRICS_CSV_FILE,
    CONFIG_JSON_FILE,
    CONFIG_YAML_FILE,
    SUMMARY_RESULTS_FILE,
    EXPERIMENT_IMAGES_TEMPLATE,
    START_EXPERIMENT_MESSAGE,
    EXPERIMENT_COMPLETED_MESSAGE,
    NO_VARIABILITY_WARNING,
    INSUFFICIENT_VARIABILITY_MESSAGE
)

class ExperimentRunner:
    def __init__(self, base_config_path, results_dir, checkpoint_interval=5):
        self.base_config_path = base_config_path
        self.results_dir = results_dir
        self.checkpoint_interval = checkpoint_interval
        self._set_seed(42)
        self.device = check_and_set_device()  # Define o dispositivo de execução (GPU ou CPU)

    @staticmethod
    def _set_seed(seed):
        """Configura a seed para garantir reprodutibilidade."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def run_experiments(self, n_experiments):
        """Executa múltiplos experimentos de treinamento GAN com hiperparâmetros variados."""
        base_config = ConfigHandler.load_yaml(self.base_config_path)
        X_train, X_valid, X_test, _, _, _ = DataLoader.load_fashion_mnist()
        train_dataset, _, _ = DataLoader.create_tf_datasets(X_train, X_valid, X_test, base_config['settings']['batch_size'])

        all_results = []
        memory_usage_list = []

        for experiment in range(n_experiments):
            print(START_EXPERIMENT_MESSAGE.format(current=experiment + 1, total=n_experiments))
            new_params = ConfigHandler.generate_new_params()
            config = {**base_config, **new_params}
            gan = GAN(config)
            evaluator = Evaluator(checkpoint_interval=self.checkpoint_interval)

            discriminator_losses, generator_losses = [], []
            
            for epoch in range(config['gan']['epochs']):
                print(f"Epoch {epoch + 1}/{config['gan']['epochs']}")
                memory_usage = os.popen("free -m | awk 'NR==2{printf $3/1024}'").read().replace(",", ".")
                memory_usage_list.append(float(memory_usage))
                disc_loss, gen_loss = gan.train_epoch(train_dataset)
                discriminator_losses.append(disc_loss)
                generator_losses.append(gen_loss)

                if (epoch + 1) % self.checkpoint_interval == 0:
                    real_images = X_test[:config['settings']['batch_size']]
                    generated_images = gan.generator(tf.random.normal(shape=[config['settings']['batch_size'], config['settings']['codings_size']])).numpy()
                    evaluator.add_metrics(real_images, generated_images, generator_losses, discriminator_losses, epoch=epoch + 1)
                    
                    # Chamada para `generate_images` com o gerador e salvar o grid de imagens
                    gan.generate_images(
                        model=gan.generator, 
                        n_images_per_row=5, 
                        n_rows=5, 
                        codings_size=config['settings']['codings_size'],
                        save_path=os.path.join(self.results_dir, EXPERIMENT_IMAGES_TEMPLATE.format(experiment=experiment + 1, epoch=epoch + 1))
                    )

            generator_losses_at_checkpoints = [loss for i, loss in enumerate(generator_losses) if (i + 1) % self.checkpoint_interval == 0]
            discriminator_losses_at_checkpoints = [loss for i, loss in enumerate(discriminator_losses) if (i + 1) % self.checkpoint_interval == 0]

            mean_ssim, lpips_score = evaluator.test_diversity(generated_images)
            evaluator.adjust_diversity_threshold(mean_ssim, lpips_score)

            if evaluator.metrics_df.empty:
                print(NO_VARIABILITY_WARNING.format(experiment=experiment + 1))
                continue

            evaluator.metrics_df['generator_loss'] = generator_losses_at_checkpoints
            evaluator.metrics_df['discriminator_loss'] = discriminator_losses_at_checkpoints
            evaluator.metrics_df['Score'] = evaluator.metrics_df['image_quality_ssim']
            evaluator.metrics_df['Ranking'] = evaluator.metrics_df['Score'].rank(ascending=False, method='min').astype(int)
            evaluator.metrics_df['Experiment'] = experiment + 1
            evaluator.metrics_df['Epoch'] = evaluator.metrics_df.index + 1
            evaluator.metrics_df['learning_rate'] = config['gan']['learning_rate']

            experiment_dir = os.path.join(self.results_dir, f"experiment_{experiment + 1}")
            os.makedirs(experiment_dir, exist_ok=True)
            ConfigHandler.save_metrics_to_csv(evaluator.metrics_df, os.path.join(experiment_dir, METRICS_CSV_FILE))
            ConfigHandler.save_json(new_params, os.path.join(experiment_dir, CONFIG_JSON_FILE))
            ConfigHandler.save_yaml(new_params, os.path.join(experiment_dir, CONFIG_YAML_FILE))

            all_results.extend(evaluator.metrics_df.to_dict(orient='records'))
            print(EXPERIMENT_COMPLETED_MESSAGE.format(experiment=experiment + 1, directory=experiment_dir))

        pd.DataFrame({'Memory Usage (GB)': memory_usage_list}).to_csv(os.path.join(self.results_dir, MEMORY_USAGE_FILE), index=False)

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df['Score'] = results_df['image_quality_ssim']
            results_df.sort_values(by='Ranking', ascending=True, inplace=True)
            results_df.to_csv(os.path.join(self.results_dir, SUMMARY_RESULTS_FILE), index=False)
            print(EXPERIMENTS_FINISHED_MESSAGE)
        else:
            print(INSUFFICIENT_VARIABILITY_MESSAGE)
