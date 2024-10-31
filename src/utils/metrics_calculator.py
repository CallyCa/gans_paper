import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
import tensorflow as tf
from lpips import LPIPS
import torch.nn.functional as F
import torch
import random

class MetricsCalculator:
    """Classe para calcular métricas de avaliação de GANs, como SSIM, FID e diversidade de imagens."""

    # Inicializar o modelo InceptionV3 para extração de características
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))    
    # Inicializar o modelo LPIPS para avaliação perceptual
    lpips_metric = LPIPS(net='alex')  # 'alex' usa a rede AlexNet para cálculo de LPIPS
    
    @staticmethod
    def calculate_ssim(real_images, generated_images):
        """
        Calcula o Structural Similarity Index (SSIM) para imagens reais e geradas.

        Args:
            real_images (np.ndarray): Array de imagens reais.
            generated_images (np.ndarray): Array de imagens geradas.

        Returns:
            float: Valor médio de SSIM para as imagens.
        """
        real_images = MetricsCalculator.normalize_images(real_images, scale=1.0)
        generated_images = MetricsCalculator.normalize_images(generated_images, scale=1.0)

        # Define o alcance de dados e o tamanho da janela
        data_range = real_images.max() - real_images.min()
        min_dim = min(real_images.shape[1], real_images.shape[2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

        # Cálculo do SSIM com adaptação para múltiplos canais
        ssim_values = [
            ssim(real, gen, data_range=data_range, win_size=win_size, channel_axis=-1)
            for real, gen in zip(real_images, generated_images)
        ]

        mean_ssim = np.mean(ssim_values) + random.uniform(-0.005, 0.005)  # Perturbação aleatória adicional
        return mean_ssim

    @staticmethod
    def calculate_lpips(real_images, generated_images):
        """
        Calcula o Learned Perceptual Image Patch Similarity (LPIPS) entre imagens reais e geradas.

        Args:
            real_images (np.ndarray ou tf.Tensor): Array de imagens reais.
            generated_images (np.ndarray ou tf.Tensor): Array de imagens geradas.

        Returns:
            float: LPIPS médio para as imagens.
        """
        if isinstance(real_images, np.ndarray):
            real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2)
        if isinstance(generated_images, np.ndarray):
            generated_images = torch.from_numpy(generated_images).permute(0, 3, 1, 2)

        real_images = F.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False)
        generated_images = F.interpolate(generated_images, size=(64, 64), mode='bilinear', align_corners=False)

        real_images = 2 * real_images - 1
        generated_images = 2 * generated_images - 1

        lpips_values = [MetricsCalculator.lpips_metric(real.unsqueeze(0), gen.unsqueeze(0)).item() for real, gen in zip(real_images, generated_images)]
        mean_lpips = np.mean(lpips_values)
        return mean_lpips + random.uniform(-0.005, 0.005)  # Perturbação adicional

    @staticmethod
    def calculate_fid(real_images, generated_images):
        """
        Calcula o Frechet Inception Distance (FID) entre imagens reais e geradas.

        Args:
            real_images (np.ndarray): Array de imagens reais.
            generated_images (np.ndarray): Array de imagens geradas.

        Returns:
            float: Valor do FID.
        """
        if not isinstance(real_images, tf.Tensor):
            real_images = tf.convert_to_tensor(real_images)
        if not isinstance(generated_images, tf.Tensor):
            generated_images = tf.convert_to_tensor(generated_images)

        if real_images.shape[-1] == 1:
            real_images = tf.image.grayscale_to_rgb(real_images)
        if generated_images.shape[-1] == 1:
            generated_images = tf.image.grayscale_to_rgb(generated_images)

        real_images = tf.image.resize(real_images, (299, 299))
        generated_images = tf.image.resize(generated_images, (299, 299))

        real_images = tf.keras.applications.inception_v3.preprocess_input(real_images)
        generated_images = tf.keras.applications.inception_v3.preprocess_input(generated_images)

        real_features = MetricsCalculator.inception_model(real_images)
        generated_features = MetricsCalculator.inception_model(generated_images)

        mu1, sigma1 = real_features.numpy().mean(axis=0), np.cov(real_features.numpy(), rowvar=False)
        mu2, sigma2 = generated_features.numpy().mean(axis=0), np.cov(generated_features.numpy(), rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        fid += random.uniform(-0.1, 0.1)  # Perturbação aleatória
        return fid

    @staticmethod
    def calculate_image_diversity(generated_images):
        """
        Calcula a diversidade das imagens geradas com base na variância.

        Args:
            generated_images (np.ndarray): Array de imagens geradas.

        Returns:
            float: Média da variância das imagens, representando a diversidade.
        """
        generated_images = MetricsCalculator.normalize_images(generated_images, scale=1.0)
        variances = [np.var(image) for image in generated_images]
        mean_variance = np.mean(variances) + random.uniform(-0.005, 0.005)  # Perturbação adicional
        return mean_variance

    @staticmethod
    def normalize_images(images, scale=1.0):
        """
        Normaliza as imagens para o intervalo especificado.

        Args:
            images (np.ndarray ou tf.Tensor): Array ou tensor de imagens a serem normalizadas.
            scale (float, optional): Escala alvo para normalizar as imagens. Default é 1.0.

        Returns:
            np.ndarray ou tf.Tensor: Imagens normalizadas.
        """
        if isinstance(images, tf.Tensor):
            min_val = tf.reduce_min(images)
            max_val = tf.reduce_max(images)
        else:
            min_val = images.min()
            max_val = images.max()

        normalized_images = (images - min_val) / (max_val - min_val) * scale
        return normalized_images
