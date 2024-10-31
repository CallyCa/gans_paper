# src/evaluation/evaluator.py

import numpy as np
import pandas as pd
from src.utils.metrics_calculator import MetricsCalculator
from gaussian_ahp.ahp_gaussian import AHPGaussian
from multiprocessing import Pool, cpu_count
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LinearRegression
from constants.constants import (
    DEFAULT_DIVERSITY_THRESHOLD,
    MIN_VARIABILITY_THRESHOLD,
    NORMALIZATION_MIN,
    NORMALIZATION_MAX,
    DEFAULT_POOL_PROCESSES,
    NO_VARIABILITY_INSUFICIENT_WARNING,
    INSUFFICIENT_VARIABILITY_ERROR,
    METRICS_CALCULATED_MESSAGE,
    AHP_RANKING_MESSAGE,
    CHECKPOINT_SAVE_MESSAGE,
    DIVERSITY_THRESHOLD_ADJUST_MESSAGE,
    LPIPS_SCORE_CALCULATION_MESSAGE
)

class Evaluator:
    """Classe para avaliação de desempenho do GAN, incluindo AHP Gaussian para análise de métricas."""

    def __init__(self, metrics_df=None, checkpoint_interval=5):
        """
        Inicializa o avaliador com um DataFrame de métricas e um threshold inicial de diversidade.

        Args:
            metrics_df (pd.DataFrame, opcional): DataFrame contendo métricas de avaliação para cada execução.
            checkpoint_interval (int): Intervalo de épocas para salvar checkpoints das métricas.
        """
        self.metrics_df = metrics_df if metrics_df is not None else pd.DataFrame()
        self.previous_ssim_scores = []
        self.previous_lpips_scores = []
        self.previous_thresholds = []
        self.diversity_threshold = DEFAULT_DIVERSITY_THRESHOLD
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []

    def calculate_ssim_pair(self, args):
        img1, img2, data_range = args
        return ssim(img1.squeeze(), img2.squeeze(), data_range=data_range)

    def test_diversity(self, generated_images):
        """Calcula o SSIM e LPIPS médio entre pares de imagens geradas para medir a diversidade."""
        data_range = generated_images.max() - generated_images.min()
        pairs = [(generated_images[i], generated_images[j], data_range) for i in range(len(generated_images)) for j in range(i + 1, len(generated_images))]

        print(LPIPS_SCORE_CALCULATION_MESSAGE)
        with Pool(processes=min(cpu_count(), DEFAULT_POOL_PROCESSES)) as pool:
            ssim_scores = pool.map(self.calculate_ssim_pair, pairs, chunksize=len(pairs) // cpu_count())
        
        lpips_score = MetricsCalculator.calculate_lpips(generated_images, generated_images)
        mean_ssim = np.mean(ssim_scores)
        self.previous_ssim_scores.append(mean_ssim)
        self.previous_lpips_scores.append(lpips_score)

        return mean_ssim, lpips_score

    def adjust_diversity_threshold(self, mean_ssim, lpips_score):
        """Ajusta o limiar de diversidade com base nos SSIMs e LPIPS anteriores."""
        if len(self.previous_ssim_scores) >= 2 and len(self.previous_lpips_scores) >= 2:
            X = np.array(list(zip(self.previous_ssim_scores, self.previous_lpips_scores)))
            y = np.array(self.previous_thresholds)
            
            model = LinearRegression()
            model.fit(X, y)
            
            predicted_threshold = model.predict(np.array([[mean_ssim, lpips_score]]))
            self.diversity_threshold = max(0.1, min(predicted_threshold[0], DEFAULT_DIVERSITY_THRESHOLD))
        else:
            self.diversity_threshold = DEFAULT_DIVERSITY_THRESHOLD
        print(DIVERSITY_THRESHOLD_ADJUST_MESSAGE.format(diversity_threshold=self.diversity_threshold))

    def add_metrics(self, real_images, generated_images, generator_losses, discriminator_losses, epoch):
        # Cálculo das métricas usando a classe MetricsCalculator
        image_quality_ssim = MetricsCalculator.calculate_ssim(real_images, generated_images)
        image_quality_fid = MetricsCalculator.calculate_fid(real_images, generated_images)
        image_diversity = MetricsCalculator.calculate_image_diversity(generated_images)
        training_stability = 1 - (np.std(generator_losses) / np.mean(generator_losses))
        overfitting_metric = abs(np.mean(generator_losses) - np.mean(discriminator_losses))
        lpips_score = MetricsCalculator.calculate_lpips(real_images, generated_images)

        # Adiciona um checkpoint para as métricas a cada `checkpoint_interval` épocas
        if epoch % self.checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'image_quality_ssim': image_quality_ssim,
                'image_quality_fid': image_quality_fid,
                'image_diversity': image_diversity,
                'training_stability': training_stability,
                'overfitting_metric': overfitting_metric,
                'lpips_score': lpips_score
            }
            self.checkpoints.append(checkpoint)
            print(CHECKPOINT_SAVE_MESSAGE)

        # Verifica variabilidade entre checkpoints recentes para decidir sobre salvamento de métricas
        if len(self.checkpoints) > 1:
            last_checkpoint = self.checkpoints[-2]
            current_variability = {
                key: abs(checkpoint[key] - last_checkpoint[key]) for key in checkpoint if key != 'epoch'
            }
            if all(value <= MIN_VARIABILITY_THRESHOLD for value in current_variability.values()):
                print(NO_VARIABILITY_INSUFICIENT_WARNING)
                return

        metrics = {
            'image_quality_ssim': image_quality_ssim,
            'image_quality_fid': image_quality_fid,
            'image_diversity': image_diversity,
            'training_stability': training_stability,
            'overfitting_metric': overfitting_metric,
            'lpips_score': lpips_score
        }
        
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics])], ignore_index=True)

    def calculate_global_preference(self):
        print(METRICS_CALCULATED_MESSAGE.format(metrics_description=self.metrics_df.describe()))

        if self.metrics_df.empty or self.metrics_df.nunique().min() <= 1:
            raise ValueError(INSUFFICIENT_VARIABILITY_ERROR)

        ahp_gaussian = AHPGaussian(self.metrics_df, objective='max')
        global_preference = ahp_gaussian.global_preference()

        print(AHP_RANKING_MESSAGE.format(global_preference=global_preference))
        return global_preference

    def normalize_metrics(self):
        """Normaliza as métricas do DataFrame para o intervalo [0, 1]."""
        normalized_df = self.metrics_df.copy()
        for column in normalized_df.columns:
            max_value = normalized_df[column].max()
            min_value = normalized_df[column].min()
            normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value) * (NORMALIZATION_MAX - NORMALIZATION_MIN) + NORMALIZATION_MIN
        return normalized_df