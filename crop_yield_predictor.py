#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop Yield Predictor - Classe Profissional para Previsão de Rendimento de Colheita
================================================================================

Esta classe implementa um preditor robusto de rendimento de colheita seguindo
padrões de engenharia de software e melhores práticas de Machine Learning.

Autor: Análise Profissional de Dados
Versão: 2.0 - Versão Profissional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from datetime import datetime
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional, Union

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configuração de logging profissional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_yield_predictor.log'),
        logging.StreamHandler()
    ]
)

class CropYieldPredictor:
    """
    Preditor profissional de rendimento de colheita com validação cruzada,
    análise de resíduos e capacidade de salvar/carregar modelos.
    
    Attributes:
        model: Modelo de Machine Learning treinado
        scaler: Normalizador de features (opcional)
        feature_columns: Lista das colunas de features
        model_metrics: Métricas de performance do modelo
    """
    
    def __init__(self, model_type: str = 'linear', normalize: bool = False, model_path: Optional[str] = None):
        """
        Inicializa o preditor de rendimento de colheita.
        
        Args:
            model_type: Tipo do modelo ('linear', 'random_forest', 'gradient_boosting')
            normalize: Se deve normalizar as features
            model_path: Caminho para carregar modelo pré-treinado
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
        # Definindo colunas de features padrão
        self.feature_columns = [
            'rainfall_mm', 'soil_quality_index', 'farm_size_hectares', 
            'sunlight_hours', 'fertilizer_kg'
        ]
        
        # Inicializando métricas
        self.model_metrics = {}
        self.training_history = []
        
        if model_path:
            self.load_model(model_path)
            self.logger.info(f"Modelo carregado de: {model_path}")
        else:
            self.model = self._create_model(model_type)
            self.logger.info(f"Novo modelo criado: {model_type}")
    
    def _create_model(self, model_type: str):
        """Cria modelo baseado no tipo especificado."""
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if model_type not in models:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}. Opções: {list(models.keys())}")
        
        return models[model_type]
    
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Carrega dados do arquivo CSV.
        
        Args:
            filepath: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Dados carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")
            
            # Validação básica dos dados
            missing_cols = [col for col in self.feature_columns + ['crop_yield'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colunas faltando no dataset: {missing_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def add_noise_simulation(self, X: pd.DataFrame, y: pd.Series, 
                           noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Adiciona ruído aos dados para testar robustez do modelo.
        
        Args:
            X: Features
            y: Target
            noise_level: Nível de ruído (proporção do desvio padrão)
            
        Returns:
            Dados com ruído adicionado
        """
        self.logger.info(f"Adicionando ruído com nível: {noise_level}")
        
        X_noisy = X.copy()
        y_noisy = y.copy()
        
        # Adicionando ruído às features numéricas
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                noise = np.random.normal(0, X[col].std() * noise_level, len(X))
                X_noisy[col] = X[col] + noise
        
        # Adicionando ruído ao target
        target_noise = np.random.normal(0, y.std() * noise_level, len(y))
        y_noisy = y + target_noise
        
        return X_noisy, y_noisy
    
    def cross_validation_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5) -> Dict[str, float]:
        """
        Realiza validação cruzada robusta do modelo.
        
        Args:
            X: Features
            y: Target
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Dicionário com métricas de validação cruzada
        """
        self.logger.info(f"Iniciando validação cruzada com {cv_folds} folds")
        
        # Preparando dados
        if self.normalize and self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            X_scaled = X
        
        # Validação cruzada
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # R² scores
        r2_scores = cross_val_score(self.model, X_scaled, y, cv=kfold, scoring='r2')
        
        # RMSE scores (negativo, então invertemos)
        rmse_scores = np.sqrt(-cross_val_score(self.model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error'))
        
        # MAE scores (negativo, então invertemos)
        mae_scores = -cross_val_score(self.model, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')
        
        cv_results = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'r2_scores': r2_scores,
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'rmse_scores': rmse_scores,
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'mae_scores': mae_scores
        }
        
        self.logger.info(f"Validação cruzada concluída:")
        self.logger.info(f"  R² médio: {cv_results['r2_mean']:.4f} (±{cv_results['r2_std']:.4f})")
        self.logger.info(f"  RMSE médio: {cv_results['rmse_mean']:.4f} (±{cv_results['rmse_std']:.4f})")
        
        return cv_results
    
    def residual_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Realiza análise de resíduos para validar assumições do modelo.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dicionário com análise de resíduos
        """
        self.logger.info("Iniciando análise de resíduos")
        
        # Preparando dados
        if self.normalize and self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            X_scaled = X
        
        # Fazendo previsões
        y_pred = self.model.predict(X_scaled)
        residuals = y - y_pred
        
        # Análise estatística dos resíduos
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': residuals.skew() if hasattr(residuals, 'skew') else np.nan,
            'residuals': residuals,
            'predictions': y_pred
        }
        
        self.logger.info(f"Análise de resíduos:")
        self.logger.info(f"  Média dos resíduos: {residual_stats['mean']:.4f}")
        self.logger.info(f"  Desvio padrão: {residual_stats['std']:.4f}")
        
        return residual_stats
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, 
              with_cross_validation: bool = True,
              with_noise_test: bool = True) -> Dict:
        """
        Treina o modelo com análises robustas de performance.
        
        Args:
            X: Features
            y: Target
            test_size: Proporção para dados de teste
            with_cross_validation: Se deve fazer validação cruzada
            with_noise_test: Se deve testar com dados com ruído
            
        Returns:
            Dicionário com métricas de treinamento
        """
        self.logger.info("Iniciando treinamento robusto do modelo")
        
        # Dividindo dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalizando se necessário
        if self.normalize and self.scaler:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Treinando modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Métricas básicas
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        basic_metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        # Análises robustas opcionais
        training_results = {'basic_metrics': basic_metrics}
        
        if with_cross_validation:
            cv_results = self.cross_validation_analysis(X_train, y_train)
            training_results['cross_validation'] = cv_results
        
        if with_noise_test:
            # Testando com ruído
            noise_levels = [0.05, 0.1, 0.2]
            noise_results = {}
            
            for noise in noise_levels:
                X_noisy, y_noisy = self.add_noise_simulation(X_train, y_train, noise)
                
                # Retreinando com dados com ruído
                temp_model = LinearRegression()  # Modelo temporário para teste
                
                if self.normalize:
                    X_noisy_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_noisy), 
                        columns=X_noisy.columns, 
                        index=X_noisy.index
                    )
                else:
                    X_noisy_scaled = X_noisy
                
                temp_model.fit(X_noisy_scaled, y_noisy)
                y_test_pred_noisy = temp_model.predict(X_test_scaled)
                
                noise_results[f'noise_{noise}'] = {
                    'r2': r2_score(y_test, y_test_pred_noisy),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_noisy))
                }
            
            training_results['noise_robustness'] = noise_results
        
        # Análise de resíduos
        residual_analysis = self.residual_analysis(X_test, y_test)
        training_results['residual_analysis'] = residual_analysis
        
        # Salvando métricas
        self.model_metrics = training_results
        
        # Histórico de treinamento
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': basic_metrics,
            'data_shape': X.shape
        }
        self.training_history.append(training_record)
        
        self.logger.info("Treinamento concluído com sucesso")
        self.logger.info(f"Performance final - R²: {basic_metrics['test_r2']:.4f}, RMSE: {basic_metrics['test_rmse']:.4f}")
        
        return training_results
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Faz previsões usando o modelo treinado.
        
        Args:
            data: Dados para previsão (DataFrame, array ou lista)
            
        Returns:
            Array com previsões
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Convertendo para DataFrame se necessário
        if isinstance(data, list):
            if len(data) != len(self.feature_columns):
                raise ValueError(f"Lista deve ter {len(self.feature_columns)} elementos")
            data = pd.DataFrame([data], columns=self.feature_columns)
        elif isinstance(data, np.ndarray):
            if data.shape[1] != len(self.feature_columns):
                raise ValueError(f"Array deve ter {len(self.feature_columns)} colunas")
            data = pd.DataFrame(data, columns=self.feature_columns)
        
        # Normalizando se necessário
        if self.normalize and self.scaler:
            data_scaled = pd.DataFrame(
                self.scaler.transform(data), 
                columns=data.columns, 
                index=data.index
            )
        else:
            data_scaled = data
        
        predictions = self.model.predict(data_scaled)
        self.logger.info(f"Previsões realizadas para {len(data)} amostras")
        
        return predictions
    
    def predict_single(self, rainfall: float, soil_quality: int, 
                      farm_size: float, sunlight: float, fertilizer: float) -> float:
        """
        Faz previsão para uma única amostra.
        
        Args:
            rainfall: Precipitação em mm
            soil_quality: Qualidade do solo (1-10)
            farm_size: Tamanho da fazenda em hectares
            sunlight: Horas de sol
            fertilizer: Quantidade de fertilizante em kg
            
        Returns:
            Rendimento previsto
        """
        data = [rainfall, soil_quality, farm_size, sunlight, fertilizer]
        prediction = self.predict(data)
        return float(prediction[0])
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Salva o modelo treinado em arquivo.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar. Treine o modelo primeiro.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'training_history': self.training_history,
            'normalize': self.normalize
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Carrega modelo de arquivo.
        
        Args:
            filepath: Caminho do modelo salvo
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            self.model_metrics = model_data.get('model_metrics', {})
            self.training_history = model_data.get('training_history', [])
            self.normalize = model_data.get('normalize', False)
            
            self.logger.info(f"Modelo carregado de: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def plot_residual_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plota análise de resíduos.
        
        Args:
            figsize: Tamanho da figura
        """
        if 'residual_analysis' not in self.model_metrics:
            raise ValueError("Análise de resíduos não disponível. Treine o modelo primeiro.")
        
        residual_data = self.model_metrics['residual_analysis']
        residuals = residual_data['residuals']
        predictions = residual_data['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🔍 ANÁLISE DE RESÍDUOS - VALIDAÇÃO DO MODELO', fontsize=16, fontweight='bold')
        
        # 1. Resíduos vs Previsões
        axes[0,0].scatter(predictions, residuals, alpha=0.6)
        axes[0,0].axhline(y=0, color='red', linestyle='--')
        axes[0,0].set_xlabel('Previsões')
        axes[0,0].set_ylabel('Resíduos')
        axes[0,0].set_title('Resíduos vs Previsões', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Histograma dos resíduos
        axes[0,1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Resíduos')
        axes[0,1].set_ylabel('Frequência')
        axes[0,1].set_title('Distribuição dos Resíduos', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot (aproximação)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
        axes[1,0].scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
        axes[1,0].plot(theoretical_quantiles, theoretical_quantiles, 'r--')
        axes[1,0].set_xlabel('Quantis Teóricos')
        axes[1,0].set_ylabel('Quantis Observados')
        axes[1,0].set_title('Q-Q Plot (Normalidade)', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Resíduos absolutos vs Previsões
        axes[1,1].scatter(predictions, np.abs(residuals), alpha=0.6, color='orange')
        axes[1,1].set_xlabel('Previsões')
        axes[1,1].set_ylabel('Resíduos Absolutos')
        axes[1,1].set_title('Resíduos Absolutos vs Previsões', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cross_validation_results(self) -> None:
        """Plota resultados da validação cruzada."""
        if 'cross_validation' not in self.model_metrics:
            raise ValueError("Validação cruzada não disponível. Treine o modelo com with_cross_validation=True.")
        
        cv_data = self.model_metrics['cross_validation']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('📊 VALIDAÇÃO CRUZADA - ESTABILIDADE DO MODELO', fontsize=16, fontweight='bold')
        
        # R² scores
        axes[0].boxplot(cv_data['r2_scores'])
        axes[0].set_title(f'R² Score\nMédia: {cv_data["r2_mean"]}')
        axes[0].set_ylabel('R²')
        
        # RMSE scores
        axes[1].boxplot(cv_data['rmse_scores'])
        axes[1].set_title(f'RMSE\nMédia: {cv_data["rmse_mean"]}')
        axes[1].set_ylabel('RMSE')
        
        # MAE scores
        axes[2].boxplot(cv_data['mae_scores'])
        axes[2].set_title(f'MAE\nMédia: {cv_data["mae_mean"]}')
        axes[2].set_ylabel('MAE')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Gera relatório completo."""
        if not self.model_metrics:
            return "Modelo não treinado"
        
        report = ["="*70, "🏆 RELATÓRIO DO MODELO", "="*70]
        
        # Métricas básicas
        basic = self.model_metrics['basic_metrics']
        report.extend([
            "\n📊 PERFORMANCE:",
            f"   R² (Teste): {basic['test_r2']:.4f} ({basic['test_r2']*100:.2f}%)",
            f"   RMSE: {basic['test_rmse']:.4f}",
            f"   MAE: {basic['test_mae']:.4f}"
        ])
        
        # Validação cruzada
        if 'cross_validation' in self.model_metrics:
            cv = self.model_metrics['cross_validation']
            report.extend([
                "\n🔄 VALIDAÇÃO CRUZADA:",
                f"   R² Médio: {cv['r2_mean']:.4f} ±{cv['r2_std']:.4f}",
                f"   RMSE Médio: {cv['rmse_mean']:.4f} ±{cv['rmse_std']:.4f}"
            ])
        
        # Robustez com ruído
        if 'noise_robustness' in self.model_metrics:
            report.append("\n🛡️ ROBUSTEZ COM RUÍDO:")
            for noise_level, metrics in self.model_metrics['noise_robustness'].items():
                level = noise_level.split('_')[1]
                report.append(f"   Ruído {level}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        # Resíduos
        if 'residual_analysis' in self.model_metrics:
            residual = self.model_metrics['residual_analysis']
            report.extend([
                "\n🔍 RESÍDUOS:",
                f"   Média: {residual['mean']:.4f}",
                f"   Desvio: {residual['std']:.4f}"
            ])
        
        report.append("="*70)
        return "\n".join(report)


# Exemplo de uso
if __name__ == "__main__":
    # Demonstração da classe
    predictor = CropYieldPredictor(model_type='linear')
    
    # Carregando e treinando
    df = predictor.load_data('crop_yield_data.csv')
    X = df[predictor.feature_columns]
    y = df['crop_yield']
    
    # Treinamento robusto
    results = predictor.train(X, y, with_cross_validation=True, with_noise_test=True)
    
    # Relatório
    print(predictor.generate_report())
    
    # Previsão
    rendimento = predictor.predict_single(1500, 8, 500, 10, 1500)
    print(f"\n🔮 Previsão: {rendimento:.2f} toneladas/hectare")
    
    # Salvando
    predictor.save_model('crop_yield_model.joblib')
    print("\n💾 Modelo salvo!")