import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ResultsAnalyzer:
    """Classe para análise e visualização de resultados dos experimentos GAN."""

    def __init__(self, results_dir):
        """
        Inicializa o analisador de resultados, carregando o arquivo de resultados de resumo.

        Args:
            results_dir (str): Caminho para o diretório onde os resultados foram armazenados.
        """
        self.results_dir = results_dir
        self.summary_file = os.path.join(results_dir, "summary_results.csv")
        if os.path.exists(self.summary_file):
            self.results_df = pd.read_csv(self.summary_file)
        else:
            raise FileNotFoundError(f"Arquivo de resultados '{self.summary_file}' não encontrado.")

    def plot_metric_distribution(self, metric):
        """
        Plota a distribuição de uma métrica específica entre os experimentos.

        Args:
            metric (str): Nome da métrica a ser plotada.
        """
        if metric not in self.results_df.columns:
            print(f"Métrica '{metric}' não encontrada no arquivo de resultados. Métricas disponíveis: {self.results_df.columns.tolist()}")
            raise ValueError(f"Métrica '{metric}' não encontrada no arquivo de resultados.")

        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results_df, x=metric, kde=True, bins=20, color="skyblue")
        plt.title(f"Distribuição da Métrica: {metric}", fontsize=14, pad=15)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel("Frequência", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def plot_hyperparam_vs_metric(self, hyperparam, metric):
        """
        Plota a relação entre um hiperparâmetro e uma métrica específica.

        Args:
            hyperparam (str): Nome do hiperparâmetro a ser comparado.
            metric (str): Nome da métrica a ser comparada.
        """
        if hyperparam not in self.results_df.columns or metric not in self.results_df.columns:
            raise ValueError(f"Hiperparâmetro '{hyperparam}' ou métrica '{metric}' não encontrado(s) no arquivo de resultados.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.results_df, x=hyperparam, y=metric, marker='o', color="salmon", edgecolor="black")
        
        # Adicionando linha de tendência
        sns.regplot(data=self.results_df, x=hyperparam, y=metric, scatter=False, color="black", line_kws={"linestyle": "--"})
        
        plt.title(f"Relação entre {hyperparam} e {metric}", fontsize=14, pad=15)
        plt.xlabel(hyperparam, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def plot_loss_convergence(self):
        """Plota a convergência das perdas do gerador e do discriminador ao longo das épocas para cada experimento."""
        plt.figure(figsize=(14, 8))
        sns.set(style="whitegrid")

        # Usando cores mais suaves e contrastantes
        generator_color = "#7B68EE"  # Medium Slate Blue
        discriminator_color = "#32CD32"  # Lime Green

        # Gráfico para as perdas do gerador e do discriminador
        sns.lineplot(
            data=self.results_df, x="Epoch", y="generator_loss", hue="Experiment", 
            marker="o", linewidth=2.5, color=generator_color, linestyle="-"
        ).set_label("Gerador")  # Define label para o Gerador

        sns.lineplot(
            data=self.results_df, x="Epoch", y="discriminator_loss", hue="Experiment", 
            marker="x", linewidth=2.5, color=discriminator_color, linestyle="--"
        ).set_label("Discriminador")  # Define label para o Discriminador

        # Ajustes dos eixos e título
        plt.xlabel("Época", fontsize=14, labelpad=10)
        plt.ylabel("Perda", fontsize=14, labelpad=10)
        plt.title("Convergência das Perdas do Gerador e Discriminador ao Longo das Épocas", fontsize=16, pad=20)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Ajusta legenda manualmente
        plt.legend(title="Tipo de Perda", fontsize=12, title_fontsize=13, loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_metrics_heatmap(self):
        """Plota um heatmap das correlações entre as métricas principais."""
        metric_cols = [col for col in self.results_df.columns if col not in ["Ranking", "Experiment", "Epoch", "Score", "learning_rate"]]
        plt.figure(figsize=(12, 8))
        sns.set(style="white")
        
        corr = self.results_df[metric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f", linewidths=1, linecolor="gray", cbar_kws={"shrink": .8})

        plt.title("Correlação Entre Métricas", fontsize=16, pad=20)
        plt.xticks(fontsize=12, rotation=45, ha="right")
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_metric_trends(self):
        """Plota as tendências de métricas específicas ao longo das épocas para cada experimento."""
        metrics_to_plot = ["image_quality_ssim", "image_quality_fid", "lpips_score", "image_diversity"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        sns.set(style="whitegrid", palette="muted")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        
        # Definir o título principal com espaçamento menor para ficar mais próximo dos subplots
        fig.suptitle("Evolução das Métricas ao Longo das Épocas", fontsize=16, y=0.95)

        for ax, metric, color in zip(axes.flat, metrics_to_plot, colors):
            sns.lineplot(
                data=self.results_df,
                x="Epoch",
                y=metric,
                hue="Experiment",
                marker="o",
                linewidth=2,
                palette=[color],
                ax=ax
            )
            ax.set_title(metric.replace('_', ' ').capitalize(), fontsize=12, pad=10)
            ax.set_xlabel("Época", fontsize=10)
            ax.set_ylabel("Valor da Métrica", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend_.remove()  # Remove legendas individuais dos subplots

        # Legenda geral para todos os subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Experimento", loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.05))

        # Ajuste do layout com espaçamento extra para o título e menor espaçamento geral
        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2, w_pad=2)
        plt.show()

    def plot_hyperparameter_distributions(self):
        """Plota a distribuição do 'learning_rate' e exibe estatísticas resumidas com formatação científica."""
        plt.figure(figsize=(12, 6))
        sns.set(style="ticks", palette="pastel")

        # Histograma com KDE
        sns.histplot(self.results_df["learning_rate"], kde=True, bins=10, color="teal", edgecolor="black", linewidth=1.2)
        plt.xlabel("Learning Rate", fontsize=14, labelpad=10)
        plt.ylabel("Frequência", fontsize=14, labelpad=10)
        plt.title("Distribuição do Learning Rate Entre Experimentos", fontsize=16, pad=20)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Estatísticas resumidas com formatação científica
        stats = self.results_df["learning_rate"].describe().apply(lambda x: f"{x:.4e}")
        stats = stats.rename(index={
            "count": "Contagem", "mean": "Média", "std": "Desvio Padrão", "min": "Mínimo",
            "25%": "1º Quartil", "50%": "Mediana", "75%": "3º Quartil", "max": "Máximo"
        })
        stats_table = pd.DataFrame(stats).transpose()

        # Adiciona a tabela de estatísticas no gráfico
        plt.table(
            cellText=stats_table.values,
            colLabels=stats_table.columns,
            rowLabels=stats_table.index,
            loc="bottom",
            cellLoc="center",
            rowLoc="center",
            bbox=[0.0, -0.35, 1, 0.25],  # Ajuste para melhor posicionamento da tabela
        )

        # Ajusta o layout para evitar sobreposição
        plt.subplots_adjust(bottom=0.25)  # Espaço extra para a tabela
        plt.show()

    def plot_all_metrics(self):
        """
        Plota todas as métricas presentes no arquivo de resultados para análise comparativa.
        """
        metrics = [col for col in self.results_df.columns if 'image_quality' in col or 'diversity' in col or 'stability' in col]
        if not metrics:
            print("Nenhuma métrica encontrada para visualização.")
            return
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.results_df[metrics], palette="Set2")
        plt.title("Comparação das Métricas de Avaliação", fontsize=14, pad=15)
        plt.ylabel("Valores das Métricas", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def plot_ranking_vs_score(self):
        if 'Ranking' not in self.results_df.columns or 'Score' not in self.results_df.columns:
            raise ValueError("Colunas 'Ranking' ou 'Score' não encontradas no arquivo de resultados.")

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-darkgrid')
        
        # Ordena e reatribui `Ranking` para garantir a visualização adequada
        self.results_df = self.results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
        self.results_df['Ranking'] = range(1, len(self.results_df) + 1)

        plt.plot(self.results_df['Ranking'], self.results_df['Score'], marker='o', linestyle='-', color='blue', label='Score by Ranking')

        for i, row in self.results_df.iterrows():
            offset_x = 10 if i % 2 == 0 else -10
            offset_y = 10 if i % 2 == 0 else -10
            ha = 'left' if offset_x > 0 else 'right'
            plt.annotate(f'Exp {row["Experiment"]}\nEpoch {row["Epoch"]}\nScore {row["Score"]:.2f}', 
                         xy=(row['Ranking'], row['Score']),
                         textcoords="offset points",
                         xytext=(offset_x, offset_y),
                         ha=ha, 
                         fontsize=8, 
                         color='darkred',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', boxstyle='round,pad=0.5'))

        plt.xlabel('Ranking', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Comparison of Scores by Ranking across Experiments and Epochs', fontsize=14, pad=20)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        # Destaques: melhor, mediano e pior score
        max_score_idx = self.results_df['Score'].idxmax()
        min_score_idx = self.results_df['Score'].idxmin()
        median_score_idx = self.results_df['Score'].sub(self.results_df['Score'].median()).abs().idxmin()

        plt.scatter(self.results_df['Ranking'][max_score_idx], self.results_df['Score'][max_score_idx], color='green', zorder=5, label='Best Score')
        plt.scatter(self.results_df['Ranking'][min_score_idx], self.results_df['Score'][min_score_idx], color='red', zorder=5, label='Worst Score')
        plt.scatter(self.results_df['Ranking'][median_score_idx], self.results_df['Score'][median_score_idx], color='blue', zorder=5, label='Median Score')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_memory_usage(memory_usage_list, epochs, save_path=None):
        """
        Plot the impact of checkpointing on memory usage during training and optionally saves it.

        Args:
            memory_usage_list (list): Lista de valores de uso de memória em cada época.
            epochs (list): Lista de épocas correspondentes ao uso de memória.
            save_path (str, opcional): Caminho para salvar o gráfico de uso de memória.
        """
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-whitegrid')
        palette = plt.get_cmap('coolwarm')

        x = np.array(epochs)
        y = np.array(memory_usage_list)

        plt.plot(x, y, color=palette(0.8), linewidth=2.5, linestyle='-', marker='o', markersize=8, 
                markeredgewidth=2, markerfacecolor='white', label='Memory Usage')

        plt.fill_between(x, y, color=palette(0.4), alpha=0.3)
        plt.title('Memory Usage During Training with Checkpointing', fontsize=15, pad=15)
        plt.xlabel('Epochs', fontsize=16, labelpad=15)
        plt.ylabel('Memory Usage (GB)', fontsize=16, labelpad=15)
        plt.grid(True, linestyle='--', alpha=0.5)

        for i in range(len(x)):
            plt.text(x[i], y[i] + 0.3, f"{y[i]:.2f} GB", ha='center', va='bottom', fontsize=10, color=palette(1.0))

        plt.ylim(0, max(memory_usage_list) + 2)
        plt.xlim(min(epochs), max(epochs))

        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight')
            print(f"Gráfico de uso de memória salvo em {save_path}")
        
        plt.show()
