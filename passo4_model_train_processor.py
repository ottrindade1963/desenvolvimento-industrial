"""
Passo 4 - Pipeline de Treinamento de Modelos
=============================================
Implementa previsão GLOBAL (agregada) e POR PAÍS (individual) para 7 modelos:
  - RandomForest (painel + por país)
  - XGBoost (painel + por país)
  - TFT/GradientBoosting (painel + por país)
  - SARIMAX (por país)
  - LSTM/MLPRegressor (por país)
  - Bayes_PartialPooling (hierárquico)
  - Bayes_CompletePooling (global)

Divisão temporal por ANO: Treino<=2016 | Val=2017-2019 | Teste>=2020
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import passo4_model_train_config as config


# ============================================================
# CLASSE: Pré-processamento de Dados
# ============================================================
class DataPreprocessor:
    """Pré-processa dados e divide por ANO (divisão temporal)."""

    def clean_and_split(self, df):
        """
        Limpa dados e divide temporalmente.
        Retorna: X_train, X_val, X_test, y_train, y_val, y_test,
                 feature_cols, country_col, year_col
        """
        year_col = 'ano' if 'ano' in df.columns else 'year'
        country_col = None
        for col in ['pais', 'country', 'pais_nome', 'country_name']:
            if col in df.columns:
                country_col = col
                break
        if country_col is None:
            for col in ['codigo_iso3', 'iso3', 'country_code']:
                if col in df.columns:
                    country_col = col
                    break

        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        feature_cols = [c for c in df.columns
                        if c not in [config.TARGET_VAR, year_col] + text_cols]

        X = df[feature_cols].copy()
        y = df[config.TARGET_VAR].copy()
        years = df[year_col]

        X = X.bfill().ffill().fillna(X.mean()).fillna(0)
        y = y.bfill().ffill().fillna(y.mean()).fillna(0)

        train_mask = years <= config.TRAIN_END_YEAR
        val_mask = (years > config.TRAIN_END_YEAR) & (years <= config.VAL_END_YEAR)
        test_mask = years > config.VAL_END_YEAR

        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
        y_train = y[train_mask].values
        y_val = y[val_mask].values
        y_test = y[test_mask].values

        print(f"  Divisao temporal por ANO:")
        print(f"    Treino (<={config.TRAIN_END_YEAR}): {len(X_train)} amostras")
        print(f"    Validacao ({config.TRAIN_END_YEAR+1}-{config.VAL_END_YEAR}): {len(X_val)} amostras")
        print(f"    Teste (>={config.VAL_END_YEAR+1}): {len(X_test)} amostras")
        print(f"    Features: {len(feature_cols)} | Target: {config.TARGET_VAR}")

        return (X_train, X_val, X_test, y_train, y_val, y_test,
                feature_cols, country_col, year_col)


# ============================================================
# CLASSE: Treinamento com Previsão Global e Por País
# ============================================================
class UnifiedModelTrainer:
    """
    Treina todos os 7 modelos com previsão GLOBAL e POR PAÍS.
    """

    def __init__(self, df, feature_cols, country_col, year_col, dataset_name, strategy_name):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.country_col = country_col
        self.year_col = year_col
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name

        if country_col and country_col in df.columns:
            self.countries = sorted(df[country_col].unique())
        else:
            self.countries = []

        self.global_metrics = {}
        self.per_country_metrics = {}
        self.models = {}
        self.predictions = {}

    def _prepare_global_data(self):
        """Prepara dados para treinamento global (painel)."""
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        feature_cols = [c for c in self.feature_cols if c in self.df.columns and c not in text_cols]

        X = self.df[feature_cols].bfill().ffill().fillna(self.df[feature_cols].mean()).fillna(0)
        y = self.df[config.TARGET_VAR].bfill().ffill().fillna(self.df[config.TARGET_VAR].mean()).fillna(0)
        years = self.df[self.year_col]

        train_mask = years <= config.TRAIN_END_YEAR
        val_mask = (years > config.TRAIN_END_YEAR) & (years <= config.VAL_END_YEAR)
        test_mask = years > config.VAL_END_YEAR

        return (X[train_mask], X[val_mask], X[test_mask],
                y[train_mask].values, y[val_mask].values, y[test_mask].values,
                feature_cols)

    def _prepare_country_data(self, country):
        """Prepara dados para um pais especifico."""
        df_c = self.df[self.df[self.country_col] == country].sort_values(self.year_col)
        text_cols = df_c.select_dtypes(include=['object', 'string']).columns.tolist()
        feature_cols = [c for c in self.feature_cols if c in df_c.columns and c not in text_cols]

        X = df_c[feature_cols].bfill().ffill().fillna(0)
        y = df_c[config.TARGET_VAR].bfill().ffill().fillna(0)
        years = df_c[self.year_col]

        train_mask = years <= config.TRAIN_END_YEAR
        val_mask = (years > config.TRAIN_END_YEAR) & (years <= config.VAL_END_YEAR)
        test_mask = years > config.VAL_END_YEAR

        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
        y_train = y[train_mask].values
        y_val = y[val_mask].values
        y_test = y[test_mask].values

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def _calc_metrics(self, y_true, y_pred):
        """Calcula R2, RMSE e MAE."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'r2': None, 'rmse': None, 'mae': None}
        try:
            return {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }
        except:
            return {'r2': None, 'rmse': None, 'mae': None}

    # ────────────────────────────────────────────────────────────
    # RANDOM FOREST
    # ────────────────────────────────────────────────────────────
    def train_random_forest(self):
        """Treina Random Forest: Global (painel) + Por Pais."""
        print(f"\n  -> Treinando Random Forest...")

        X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = self._prepare_global_data()

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        search = RandomizedSearchCV(rf, config.GRID_RANDOMFOREST, n_iter=30,
                                     cv=3, scoring='neg_mean_squared_error',
                                     random_state=config.RANDOM_STATE, n_jobs=-1)
        search.fit(X_tr_s, y_train)
        best_model = search.best_estimator_

        val_preds = best_model.predict(X_val_s)
        test_preds = best_model.predict(X_test_s)

        self.global_metrics['RandomForest'] = {
            'val': self._calc_metrics(y_val, val_preds),
            'test': self._calc_metrics(y_test, test_preds),
            'best_params': search.best_params_
        }

        print(f"     [GLOBAL] Val R2={self.global_metrics['RandomForest']['val']['r2']:.4f} | "
              f"RMSE={self.global_metrics['RandomForest']['val']['rmse']:.4f} | "
              f"MAE={self.global_metrics['RandomForest']['val']['mae']:.4f}")

        # Previsao POR PAIS
        self.per_country_metrics['RandomForest'] = {}
        self.predictions['RandomForest'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}

        if self.countries:
            for country in self.countries:
                try:
                    X_tr_c, X_val_c, _, y_tr_c, y_val_c, _, _ = self._prepare_country_data(country)
                    if len(y_val_c) < 1 or len(y_tr_c) < 5:
                        continue
                    X_val_c_s = scaler.transform(X_val_c)
                    preds_c = best_model.predict(X_val_c_s)
                    metrics_c = self._calc_metrics(y_val_c, preds_c)
                    self.per_country_metrics['RandomForest'][country] = metrics_c
                    self.predictions['RandomForest']['per_country'][country] = {'y_true': y_val_c, 'y_pred': preds_c}
                except:
                    continue

            valid_r2 = [m['r2'] for m in self.per_country_metrics['RandomForest'].values() if m['r2'] is not None]
            if valid_r2:
                print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                      f"Mediana R2={np.median(valid_r2):.4f} | "
                      f"Paises: {len(valid_r2)}/{len(self.countries)}")

        self.models['RandomForest'] = {'model': best_model, 'scaler': scaler}

    # ────────────────────────────────────────────────────────────
    # XGBOOST
    # ────────────────────────────────────────────────────────────
    def train_xgboost(self):
        """Treina XGBoost: Global (painel) + Por Pais."""
        print(f"\n  -> Treinando XGBoost...")

        X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = self._prepare_global_data()

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        xgb_model = xgb.XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        search = RandomizedSearchCV(xgb_model, config.GRID_XGBOOST, n_iter=30,
                                     cv=3, scoring='neg_mean_squared_error',
                                     random_state=config.RANDOM_STATE, n_jobs=-1)
        search.fit(X_tr_s, y_train)
        best_model = search.best_estimator_

        val_preds = best_model.predict(X_val_s)
        test_preds = best_model.predict(X_test_s)

        self.global_metrics['XGBoost'] = {
            'val': self._calc_metrics(y_val, val_preds),
            'test': self._calc_metrics(y_test, test_preds),
            'best_params': search.best_params_
        }

        print(f"     [GLOBAL] Val R2={self.global_metrics['XGBoost']['val']['r2']:.4f} | "
              f"RMSE={self.global_metrics['XGBoost']['val']['rmse']:.4f} | "
              f"MAE={self.global_metrics['XGBoost']['val']['mae']:.4f}")

        self.per_country_metrics['XGBoost'] = {}
        self.predictions['XGBoost'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}

        if self.countries:
            for country in self.countries:
                try:
                    X_tr_c, X_val_c, _, y_tr_c, y_val_c, _, _ = self._prepare_country_data(country)
                    if len(y_val_c) < 1 or len(y_tr_c) < 5:
                        continue
                    X_val_c_s = scaler.transform(X_val_c)
                    preds_c = best_model.predict(X_val_c_s)
                    metrics_c = self._calc_metrics(y_val_c, preds_c)
                    self.per_country_metrics['XGBoost'][country] = metrics_c
                    self.predictions['XGBoost']['per_country'][country] = {'y_true': y_val_c, 'y_pred': preds_c}
                except:
                    continue

            valid_r2 = [m['r2'] for m in self.per_country_metrics['XGBoost'].values() if m['r2'] is not None]
            if valid_r2:
                print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                      f"Mediana R2={np.median(valid_r2):.4f} | "
                      f"Paises: {len(valid_r2)}/{len(self.countries)}")

        self.models['XGBoost'] = {'model': best_model, 'scaler': scaler}

    # ────────────────────────────────────────────────────────────
    # TFT (GradientBoosting proxy)
    # ────────────────────────────────────────────────────────────
    def train_tft(self):
        """Treina TFT (GradientBoosting proxy): Global (painel) + Por Pais."""
        print(f"\n  -> Treinando TFT (GradientBoosting)...")

        X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = self._prepare_global_data()

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        gb = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
        search = RandomizedSearchCV(gb, config.GRID_TFT, n_iter=30,
                                     cv=3, scoring='neg_mean_squared_error',
                                     random_state=config.RANDOM_STATE, n_jobs=-1)
        search.fit(X_tr_s, y_train)
        best_model = search.best_estimator_

        val_preds = best_model.predict(X_val_s)
        test_preds = best_model.predict(X_test_s)

        self.global_metrics['TFT'] = {
            'val': self._calc_metrics(y_val, val_preds),
            'test': self._calc_metrics(y_test, test_preds),
            'best_params': search.best_params_
        }

        print(f"     [GLOBAL] Val R2={self.global_metrics['TFT']['val']['r2']:.4f} | "
              f"RMSE={self.global_metrics['TFT']['val']['rmse']:.4f} | "
              f"MAE={self.global_metrics['TFT']['val']['mae']:.4f}")

        self.per_country_metrics['TFT'] = {}
        self.predictions['TFT'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}

        if self.countries:
            for country in self.countries:
                try:
                    X_tr_c, X_val_c, _, y_tr_c, y_val_c, _, _ = self._prepare_country_data(country)
                    if len(y_val_c) < 1 or len(y_tr_c) < 5:
                        continue
                    X_val_c_s = scaler.transform(X_val_c)
                    preds_c = best_model.predict(X_val_c_s)
                    metrics_c = self._calc_metrics(y_val_c, preds_c)
                    self.per_country_metrics['TFT'][country] = metrics_c
                    self.predictions['TFT']['per_country'][country] = {'y_true': y_val_c, 'y_pred': preds_c}
                except:
                    continue

            valid_r2 = [m['r2'] for m in self.per_country_metrics['TFT'].values() if m['r2'] is not None]
            if valid_r2:
                print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                      f"Mediana R2={np.median(valid_r2):.4f} | "
                      f"Paises: {len(valid_r2)}/{len(self.countries)}")

        self.models['TFT'] = {'model': best_model, 'scaler': scaler}

    # ────────────────────────────────────────────────────────────
    # SARIMAX (por pais + global agregado)
    # ────────────────────────────────────────────────────────────
    def train_sarimax(self):
        """Treina SARIMAX por pais e calcula metricas globais agregadas."""
        print(f"\n  -> Treinando SARIMAX (por pais)...")

        if not self.countries:
            print(f"     Sem coluna de pais detectada. Pulando SARIMAX.")
            return

        self.per_country_metrics['SARIMAX'] = {}
        self.predictions['SARIMAX'] = {'global': {'y_true': [], 'y_pred': []}, 'per_country': {}}
        all_val_true = []
        all_val_pred = []
        countries_trained = 0

        for country in self.countries:
            try:
                X_tr_c, X_val_c, _, y_tr_c, y_val_c, _, feat_cols = self._prepare_country_data(country)

                if len(y_tr_c) < 10 or len(y_val_c) < 1:
                    continue

                n_exog = min(3, len(feat_cols))
                corr_df = X_tr_c.copy()
                corr_df['__target__'] = y_tr_c
                corr = corr_df.corr()['__target__'].drop('__target__').abs().sort_values(ascending=False)
                top_feat = corr.head(n_exog).index.tolist()

                exog_tr = X_tr_c[top_feat].values
                exog_val = X_val_c[top_feat].values

                try:
                    p_val = adfuller(pd.Series(y_tr_c).dropna(), autolag='AIC')[1]
                    d = 0 if p_val < 0.05 else 1
                except:
                    d = 1

                model = SARIMAX(endog=y_tr_c, exog=exog_tr, order=(1, d, 1),
                               seasonal_order=(0, 0, 0, 0),
                               enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit(disp=False, maxiter=200)
                val_preds = fitted.forecast(steps=len(y_val_c), exog=exog_val)

                if np.any(np.isnan(val_preds)) or np.any(np.isinf(val_preds)):
                    continue

                metrics_c = self._calc_metrics(y_val_c, val_preds)
                self.per_country_metrics['SARIMAX'][country] = metrics_c
                self.predictions['SARIMAX']['per_country'][country] = {'y_true': y_val_c, 'y_pred': val_preds}

                all_val_true.extend(y_val_c.tolist())
                all_val_pred.extend(val_preds.tolist())
                countries_trained += 1

            except:
                continue

        if all_val_true:
            self.global_metrics['SARIMAX'] = {
                'val': self._calc_metrics(np.array(all_val_true), np.array(all_val_pred)),
                'test': {'r2': None, 'rmse': None, 'mae': None},
                'countries_trained': countries_trained
            }
            self.predictions['SARIMAX']['global'] = {'y_true': np.array(all_val_true), 'y_pred': np.array(all_val_pred)}

            print(f"     [GLOBAL] Val R2={self.global_metrics['SARIMAX']['val']['r2']:.4f} | "
                  f"RMSE={self.global_metrics['SARIMAX']['val']['rmse']:.4f} | "
                  f"MAE={self.global_metrics['SARIMAX']['val']['mae']:.4f}")

            valid_r2 = [m['r2'] for m in self.per_country_metrics['SARIMAX'].values() if m['r2'] is not None]
            if valid_r2:
                print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                      f"Mediana R2={np.median(valid_r2):.4f} | "
                      f"Paises: {countries_trained}/{len(self.countries)}")

    # ────────────────────────────────────────────────────────────
    # LSTM (MLPRegressor por pais + global agregado)
    # ────────────────────────────────────────────────────────────
    def train_lstm(self):
        """Treina LSTM (MLPRegressor) por pais e calcula metricas globais agregadas."""
        print(f"\n  -> Treinando LSTM/MLP (por pais, max_iter=5000)...")

        if not self.countries:
            print(f"     Sem coluna de pais detectada. Pulando LSTM.")
            return

        self.per_country_metrics['LSTM'] = {}
        self.predictions['LSTM'] = {'global': {'y_true': [], 'y_pred': []}, 'per_country': {}}
        all_val_true = []
        all_val_pred = []
        countries_trained = 0

        for country in self.countries:
            try:
                X_tr_c, X_val_c, _, y_tr_c, y_val_c, _, feat_cols = self._prepare_country_data(country)

                if len(y_tr_c) < 10 or len(y_val_c) < 1:
                    continue

                scaler = MinMaxScaler()
                X_tr_s = scaler.fit_transform(X_tr_c)
                X_val_s = scaler.transform(X_val_c)

                mlp = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu', solver='adam', alpha=0.001,
                    learning_rate='adaptive', max_iter=5000,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=30,
                    batch_size=min(32, max(1, len(X_tr_s) // 2)),
                    random_state=config.RANDOM_STATE
                )
                mlp.fit(X_tr_s, y_tr_c)
                val_preds = mlp.predict(X_val_s)

                if np.any(np.isnan(val_preds)) or np.any(np.isinf(val_preds)):
                    continue

                metrics_c = self._calc_metrics(y_val_c, val_preds)
                self.per_country_metrics['LSTM'][country] = metrics_c
                self.predictions['LSTM']['per_country'][country] = {'y_true': y_val_c, 'y_pred': val_preds}

                all_val_true.extend(y_val_c.tolist())
                all_val_pred.extend(val_preds.tolist())
                countries_trained += 1

            except:
                continue

        if all_val_true:
            self.global_metrics['LSTM'] = {
                'val': self._calc_metrics(np.array(all_val_true), np.array(all_val_pred)),
                'test': {'r2': None, 'rmse': None, 'mae': None},
                'countries_trained': countries_trained
            }
            self.predictions['LSTM']['global'] = {'y_true': np.array(all_val_true), 'y_pred': np.array(all_val_pred)}

            print(f"     [GLOBAL] Val R2={self.global_metrics['LSTM']['val']['r2']:.4f} | "
                  f"RMSE={self.global_metrics['LSTM']['val']['rmse']:.4f} | "
                  f"MAE={self.global_metrics['LSTM']['val']['mae']:.4f}")

            valid_r2 = [m['r2'] for m in self.per_country_metrics['LSTM'].values() if m['r2'] is not None]
            if valid_r2:
                print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                      f"Mediana R2={np.median(valid_r2):.4f} | "
                      f"Paises: {countries_trained}/{len(self.countries)}")

    # ────────────────────────────────────────────────────────────
    # BAYESIANOS (3 Estrategias de Pooling)
    # ────────────────────────────────────────────────────────────
    def train_bayesian_all(self):
        """Treina 2 modelos Bayesianos: Partial Pooling + Complete Pooling."""
        try:
            from passo4_bayesian_model import train_all_bayesian_models

            bayes_models = train_all_bayesian_models(
                self.df, self.feature_cols, self.country_col, self.year_col
            )

            for bayes_name, bayes_model in bayes_models.items():
                val_mask = (self.df[self.year_col] > config.TRAIN_END_YEAR) & \
                           (self.df[self.year_col] <= config.VAL_END_YEAR)
                test_mask = self.df[self.year_col] > config.VAL_END_YEAR
                
                df_val = self.df[val_mask].copy()
                df_test = self.df[test_mask].copy()
                
                val_preds, val_idx = bayes_model.predict(df_val, self.feature_cols, self.country_col)
                test_preds, test_idx = bayes_model.predict(df_test, self.feature_cols, self.country_col)
                
                if len(val_preds) == 0:
                    print(f"     \u26a0\ufe0f {bayes_name}: Sem previsoes de validacao")
                    continue
                
                # Obter y_true e filtrar NaN no target
                y_val_raw = df_val.loc[val_idx, config.TARGET_VAR].values
                valid_mask_val = ~np.isnan(y_val_raw)
                y_val = y_val_raw[valid_mask_val]
                val_preds = val_preds[valid_mask_val]
                
                if len(test_preds) > 0:
                    y_test_raw = df_test.loc[test_idx, config.TARGET_VAR].values
                    valid_mask_test = ~np.isnan(y_test_raw)
                    y_test = y_test_raw[valid_mask_test]
                    test_preds = test_preds[valid_mask_test]
                else:
                    y_test = np.array([])
                
                val_metrics = self._calc_metrics(y_val, val_preds)
                test_metrics = self._calc_metrics(y_test, test_preds) if len(test_preds) > 0 else {}
                
                self.global_metrics[bayes_name] = {
                    'val': val_metrics,
                    'test': test_metrics
                }
                
                r2_str = f"{val_metrics['r2']:.4f}" if val_metrics.get('r2') is not None else "N/A"
                rmse_str = f"{val_metrics['rmse']:.4f}" if val_metrics.get('rmse') is not None else "N/A"
                mae_str = f"{val_metrics['mae']:.4f}" if val_metrics.get('mae') is not None else "N/A"
                print(f"     [GLOBAL] Val R2={r2_str} | RMSE={rmse_str} | MAE={mae_str}")
                
                self.per_country_metrics[bayes_name] = {}
                self.predictions[bayes_name] = {
                    'global': {'y_true': y_val, 'y_pred': val_preds},
                    'per_country': {}
                }
                
                for country in self.countries:
                    try:
                        df_val_c = df_val[df_val[self.country_col] == country].copy()
                        if len(df_val_c) < 1:
                            continue
                        preds_c, idx_c = bayes_model.predict(df_val_c, self.feature_cols, self.country_col)
                        if len(preds_c) == 0:
                            continue
                        y_val_c_raw = df_val_c.loc[idx_c, config.TARGET_VAR].values
                        valid_c = ~np.isnan(y_val_c_raw)
                        y_val_c = y_val_c_raw[valid_c]
                        preds_c = preds_c[valid_c]
                        if len(y_val_c) == 0:
                            continue
                        metrics_c = self._calc_metrics(y_val_c, preds_c)
                        self.per_country_metrics[bayes_name][country] = metrics_c
                        self.predictions[bayes_name]['per_country'][country] = {
                            'y_true': y_val_c, 'y_pred': preds_c
                        }
                    except:
                        continue
                
                valid_r2 = [m['r2'] for m in self.per_country_metrics[bayes_name].values() if m.get('r2') is not None]
                if valid_r2:
                    print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                          f"Mediana R2={np.median(valid_r2):.4f} | "
                          f"Paises: {len(valid_r2)}/{len(self.countries)} | "
                          f"R2>0: {sum(1 for r in valid_r2 if r > 0)}")
                
                self.models[bayes_name] = {'model': bayes_model}

        except Exception as e:
            print(f"     \u274c Erro nos modelos Bayesianos: {e}")
            import traceback
            traceback.print_exc()

    # ────────────────────────────────────────────────────────────
    # TREINAR TODOS
    # ────────────────────────────────────────────────────────────
    def train_all(self):
        """Treina todos os 7 modelos (5 classicos + 2 Bayesianos)."""
        self.train_random_forest()
        self.train_xgboost()
        self.train_tft()
        self.train_sarimax()
        self.train_lstm()
        self.train_bayesian_all()

    def get_summary(self):
        """Retorna resumo dos resultados."""
        summary = []
        all_models = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
                      'Bayes_PartialPooling', 'Bayes_CompletePooling']
        for model_name in all_models:
            if model_name in self.global_metrics:
                gm = self.global_metrics[model_name]['val']
                gm_test = self.global_metrics[model_name].get('test', {})
                per_country = self.per_country_metrics.get(model_name, {})
                valid_r2 = [m['r2'] for m in per_country.values() if m.get('r2') is not None]

                summary.append({
                    'Dataset': self.dataset_name,
                    'Estrategia': self.strategy_name,
                    'Modelo': model_name,
                    'Global_R2': gm.get('r2'),
                    'Global_RMSE': gm.get('rmse'),
                    'Global_MAE': gm.get('mae'),
                    'Test_R2': gm_test.get('r2') if gm_test else None,
                    'Test_RMSE': gm_test.get('rmse') if gm_test else None,
                    'Test_MAE': gm_test.get('mae') if gm_test else None,
                    'PerCountry_Mean_R2': np.mean(valid_r2) if valid_r2 else None,
                    'PerCountry_Median_R2': np.median(valid_r2) if valid_r2 else None,
                    'N_Countries': len(valid_r2),
                    'N_Countries_R2_Positive': sum(1 for r in valid_r2 if r > 0)
                })
        return summary

    def save_results(self, output_dir=None):
        """Salva modelos, previsoes e metricas."""
        if output_dir is None:
            output_dir = config.OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{self.dataset_name}_{self.strategy_name}"

        # 1. Salvar modelos (.pkl)
        for model_name, model_data in self.models.items():
            model_path = os.path.join(output_dir, f"{prefix}_{model_name}.pkl")
            try:
                if 'Bayes' in model_name and isinstance(model_data, dict):
                    save_data = {'model_type': model_name}
                    for k, v in model_data.items():
                        if hasattr(v, 'feature_names'):
                            save_data[k + '_info'] = {
                                'feature_names': v.feature_names,
                                'countries': v.countries,
                                'strategy_type': v.strategy_type
                            }
                    with open(model_path, 'wb') as f:
                        pickle.dump(save_data, f)
                else:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)
            except Exception as e:
                print(f"     Nao foi possivel salvar {model_name}: {e}")

        # 2. Salvar METRICAS GLOBAIS em CSV
        global_rows = []
        all_models = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
                      'Bayes_PartialPooling', 'Bayes_CompletePooling']
        for model_name in all_models:
            if model_name in self.global_metrics:
                gm_val = self.global_metrics[model_name].get('val', {})
                gm_test = self.global_metrics[model_name].get('test', {})
                per_country = self.per_country_metrics.get(model_name, {})
                valid_r2 = [m['r2'] for m in per_country.values() if m.get('r2') is not None]

                global_rows.append({
                    'Dataset': self.dataset_name,
                    'Estrategia': self.strategy_name,
                    'Modelo': model_name,
                    'Val_R2': gm_val.get('r2'),
                    'Val_RMSE': gm_val.get('rmse'),
                    'Val_MAE': gm_val.get('mae'),
                    'Test_R2': gm_test.get('r2') if gm_test else None,
                    'Test_RMSE': gm_test.get('rmse') if gm_test else None,
                    'Test_MAE': gm_test.get('mae') if gm_test else None,
                    'PerCountry_Mean_R2': np.mean(valid_r2) if valid_r2 else None,
                    'PerCountry_Median_R2': np.median(valid_r2) if valid_r2 else None,
                    'PerCountry_Mean_RMSE': np.mean([m['rmse'] for m in per_country.values() if m.get('rmse') is not None]) if per_country else None,
                    'PerCountry_Mean_MAE': np.mean([m['mae'] for m in per_country.values() if m.get('mae') is not None]) if per_country else None,
                    'N_Countries': len(valid_r2),
                    'N_Countries_R2_Positive': sum(1 for r in valid_r2 if r > 0)
                })

        if global_rows:
            df_global = pd.DataFrame(global_rows)
            global_csv_path = os.path.join(output_dir, f"{prefix}_metricas_globais.csv")
            df_global.to_csv(global_csv_path, index=False)

        # 3. Salvar metricas POR PAIS em CSV
        for model_name, country_metrics in self.per_country_metrics.items():
            if country_metrics:
                rows = []
                for country, metrics in country_metrics.items():
                    rows.append({
                        'Dataset': self.dataset_name,
                        'Estrategia': self.strategy_name,
                        'Modelo': model_name,
                        'Pais': country,
                        'R2': metrics.get('r2'),
                        'RMSE': metrics.get('rmse'),
                        'MAE': metrics.get('mae')
                    })
                df_metrics = pd.DataFrame(rows)
                csv_path = os.path.join(output_dir, f"{prefix}_{model_name}_metricas_por_pais.csv")
                df_metrics.to_csv(csv_path, index=False)

        # 4. Salvar previsoes (.pkl)
        predictions_path = os.path.join(output_dir, f"{prefix}_predictions.pkl")
        with open(predictions_path, 'wb') as f:
            pickle.dump(self.predictions, f)

        # 5. Salvar training_logs para o visualizer
        logs_path = os.path.join(output_dir, 'training_logs.pkl')
        logs = {
            'global_metrics': self.global_metrics,
            'per_country_metrics': self.per_country_metrics,
            'dataset': self.dataset_name,
            'strategy': self.strategy_name
        }
        with open(logs_path, 'wb') as f:
            pickle.dump(logs, f)


# ============================================================
# FUNCAO PRINCIPAL: run_training_for_all
# ============================================================
def run_training_for_all():
    """
    Executa o treinamento completo para todos os datasets e estrategias.
    Gera previsao GLOBAL e POR PAIS para todos os 7 modelos.
    No final, imprime tabelas comparativas completas.
    """
    print("\n" + "=" * 90)
    print("INICIANDO TREINAMENTO COMPLETO DE MODELOS")
    print("  Modelos Classicos: RF, XGBoost, TFT, SARIMAX, LSTM")
    print("  Modelos Bayesianos: PartialPooling, CompletePooling")
    print("  Metricas: GLOBAL (agregada) + POR PAIS (individual)")
    print("=" * 90)

    # Definir datasets e estrategias
    datasets_strategies = []
    for dataset in config.DATASETS:
        if dataset == 'nao_agregado':
            datasets_strategies.append(('nao_agregado', 'A1_Direta'))
        else:
            for strategy in config.STRATEGIES:
                datasets_strategies.append((dataset, strategy))

    # Carregar datasets
    loaded_data = {}
    for dataset_name, strategy_name in datasets_strategies:
        filename = f"{dataset_name}_{strategy_name}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            loaded_data[(dataset_name, strategy_name)] = df
            print(f"\n  Carregando: {filename}")
            print(f"     -> {df.shape[0]} linhas x {df.shape[1]} colunas")
        else:
            print(f"\n  Dataset nao encontrado: {filepath}")

    if not loaded_data:
        print(f"\n  Nenhum dataset encontrado em {config.DATA_DIR}/")
        print(f"  Certifique-se de executar o Passo 3 primeiro!")
        return

    # Treinar para cada dataset/estrategia
    all_summaries = []

    for (dataset_name, strategy_name), df in loaded_data.items():
        print(f"\n  {'=' * 70}")
        print(f"  Treinando: {dataset_name} x {strategy_name}")
        print(f"  {'=' * 70}")

        preprocessor = DataPreprocessor()
        _, _, _, _, _, _, feature_cols, country_col, year_col = preprocessor.clean_and_split(df)

        trainer = UnifiedModelTrainer(df, feature_cols, country_col, year_col, dataset_name, strategy_name)
        trainer.train_all()
        trainer.save_results()
        all_summaries.extend(trainer.get_summary())

    # ================================================================
    # RESUMO FINAL - TABELAS COMPARATIVAS COMPLETAS
    # ================================================================
    if not all_summaries:
        print("\n  Nenhum resultado para reportar.")
        return

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(config.OUTPUT_DIR, 'resumo_treinamento_completo.csv')
    summary_df.to_csv(summary_path, index=False)

    # ─── TABELA 1: RESUMO GERAL ───
    print(f"\n\n{'=' * 100}")
    print(f"{'TABELA 1: RESUMO GERAL - METRICAS GLOBAIS (TODOS OS MODELOS)':^100}")
    print(f"{'=' * 100}")
    header = f"  {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'R2 Val':<10} {'RMSE Val':<10} {'MAE Val':<10} {'Pais Med.R2':<12} {'Paises R2>0':<12}"
    print(header)
    print(f"  {'-' * 96}")
    for _, row in summary_df.sort_values('Global_R2', ascending=False).iterrows():
        gr2 = f"{row['Global_R2']:.4f}" if pd.notna(row['Global_R2']) else "N/A"
        grmse = f"{row['Global_RMSE']:.4f}" if pd.notna(row['Global_RMSE']) else "N/A"
        gmae = f"{row['Global_MAE']:.4f}" if pd.notna(row['Global_MAE']) else "N/A"
        pr2 = f"{row['PerCountry_Median_R2']:.4f}" if pd.notna(row.get('PerCountry_Median_R2')) else "N/A"
        pos = f"{int(row['N_Countries_R2_Positive'])}/{int(row['N_Countries'])}" if pd.notna(row.get('N_Countries')) else "N/A"
        print(f"  {row['Dataset']:<14} {row['Estrategia']:<14} {row['Modelo']:<22} {gr2:<10} {grmse:<10} {gmae:<10} {pr2:<12} {pos:<12}")

    # ─── TABELA 2: RANKING TOP 20 ───
    valid = summary_df[summary_df['Global_R2'].notna()].copy()
    if not valid.empty:
        top20 = valid.nlargest(20, 'Global_R2')
        print(f"\n\n{'=' * 100}")
        print(f"{'TABELA 2: RANKING TOP 20 - MELHORES COMBINACOES':^100}")
        print(f"{'=' * 100}")
        print(f"  {'#':<4} {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'R2':<10} {'RMSE':<10} {'MAE':<10}")
        print(f"  {'-' * 82}")
        for rank, (_, row) in enumerate(top20.iterrows(), 1):
            print(f"  {rank:<4} {row['Dataset']:<14} {row['Estrategia']:<14} {row['Modelo']:<22} "
                  f"{row['Global_R2']:.4f}     {row['Global_RMSE']:.4f}     {row['Global_MAE']:.4f}")

    # ─── TABELA 3: COMPARACAO BAYESIANOS vs CLASSICOS ───
    if not valid.empty:
        bayes_models = valid[valid['Modelo'].str.contains('Bayes')]
        classic_models = valid[~valid['Modelo'].str.contains('Bayes')]

        print(f"\n\n{'=' * 100}")
        print(f"{'TABELA 3: COMPARACAO BAYESIANOS vs CLASSICOS':^100}")
        print(f"{'=' * 100}")

        print(f"\n  --- Melhores Modelos CLASSICOS (por dataset) ---")
        print(f"  {'Dataset':<14} {'Estrat.':<14} {'Modelo':<16} {'R2':<10} {'RMSE':<10} {'Pais Med.R2':<12}")
        print(f"  {'-' * 74}")
        if not classic_models.empty:
            for ds in sorted(classic_models['Dataset'].unique()):
                ds_data = classic_models[classic_models['Dataset'] == ds]
                best = ds_data.loc[ds_data['Global_R2'].idxmax()]
                pr2 = f"{best['PerCountry_Median_R2']:.4f}" if pd.notna(best.get('PerCountry_Median_R2')) else "N/A"
                print(f"  {best['Dataset']:<14} {best['Estrategia']:<14} {best['Modelo']:<16} {best['Global_R2']:.4f}     {best['Global_RMSE']:.4f}     {pr2:<12}")

        print(f"\n  --- Melhores Modelos BAYESIANOS (por dataset) ---")
        print(f"  {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'R2':<10} {'RMSE':<10} {'Pais Med.R2':<12}")
        print(f"  {'-' * 80}")
        if not bayes_models.empty:
            for ds in sorted(bayes_models['Dataset'].unique()):
                ds_data = bayes_models[bayes_models['Dataset'] == ds]
                best = ds_data.loc[ds_data['Global_R2'].idxmax()]
                pr2 = f"{best['PerCountry_Median_R2']:.4f}" if pd.notna(best.get('PerCountry_Median_R2')) else "N/A"
                print(f"  {best['Dataset']:<14} {best['Estrategia']:<14} {best['Modelo']:<22} {best['Global_R2']:.4f}     {best['Global_RMSE']:.4f}     {pr2:<12}")

    # ─── TABELA 4: COMPARACAO DAS 2 ESTRATEGIAS BAYESIANAS ───
    if not valid.empty:
        bayes_models = valid[valid['Modelo'].str.contains('Bayes')]
        if not bayes_models.empty:
            print(f"\n\n{'=' * 100}")
            print(f"{'TABELA 4: COMPARACAO DAS 2 ESTRATEGIAS BAYESIANAS':^100}")
            print(f"{'=' * 100}")

            print(f"\n  {'Dataset':<14} {'Estrat.':<14} {'PartialPool':<14} {'CompletePool':<14} {'Melhor':<20}")
            print(f"  {'-' * 74}")

            for ds in sorted(bayes_models['Dataset'].unique()):
                for est in sorted(bayes_models[bayes_models['Dataset'] == ds]['Estrategia'].unique()):
                    subset = bayes_models[(bayes_models['Dataset'] == ds) & (bayes_models['Estrategia'] == est)]
                    vals = {}
                    for _, row in subset.iterrows():
                        vals[row['Modelo']] = row['Global_R2']

                    pp = f"{vals.get('Bayes_PartialPooling', float('nan')):.4f}" if pd.notna(vals.get('Bayes_PartialPooling')) else "N/A"
                    cp = f"{vals.get('Bayes_CompletePooling', float('nan')):.4f}" if pd.notna(vals.get('Bayes_CompletePooling')) else "N/A"

                    valid_vals = {k: v for k, v in vals.items() if pd.notna(v)}
                    best_name = max(valid_vals, key=valid_vals.get).replace('Bayes_', '') if valid_vals else "N/A"

                    print(f"  {ds:<14} {est:<14} {pp:<14} {cp:<14} {best_name:<20}")

    # ─── TABELA 5: GANHO PREDITIVO ───
    if not valid.empty:
        baseline = valid[valid['Dataset'] == 'nao_agregado']
        aggregated = valid[valid['Dataset'] != 'nao_agregado']

        if not baseline.empty and not aggregated.empty:
            best_baseline = baseline.loc[baseline['Global_R2'].idxmax()]
            best_agg = aggregated.loc[aggregated['Global_R2'].idxmax()]
            best_overall = valid.loc[valid['Global_R2'].idxmax()]

            print(f"\n\n{'=' * 100}")
            print(f"{'TABELA 5: GANHO PREDITIVO':^100}")
            print(f"{'=' * 100}")
            print(f"\n  {'Metrica':<50} {'Valor':<50}")
            print(f"  {'-' * 98}")
            print(f"  {'Baseline (nao_agregado, melhor modelo)':<50} R2 = {best_baseline['Global_R2']:.4f} ({best_baseline['Modelo']})")
            print(f"  {'Melhor Agregado':<50} R2 = {best_agg['Global_R2']:.4f} ({best_agg['Dataset']} x {best_agg['Estrategia']} x {best_agg['Modelo']})")
            print(f"  {'Melhor Global (qualquer dataset)':<50} R2 = {best_overall['Global_R2']:.4f} ({best_overall['Dataset']} x {best_overall['Estrategia']} x {best_overall['Modelo']})")

            ganho = best_agg['Global_R2'] - best_baseline['Global_R2']
            ganho_pct = (ganho / abs(best_baseline['Global_R2'])) * 100 if best_baseline['Global_R2'] != 0 else 0
            print(f"  {'Ganho Absoluto (Agregacao vs Baseline)':<50} {ganho:+.4f}")
            print(f"  {'Ganho Relativo (Agregacao vs Baseline)':<50} {ganho_pct:+.1f}%")

            # Ganho Bayesiano vs Classico
            bayes_m = valid[valid['Modelo'].str.contains('Bayes')]
            classic_m = valid[~valid['Modelo'].str.contains('Bayes')]
            if not classic_m.empty and not bayes_m.empty:
                best_classic = classic_m.loc[classic_m['Global_R2'].idxmax()]
                best_bayes = bayes_m.loc[bayes_m['Global_R2'].idxmax()]
                ganho_b = best_bayes['Global_R2'] - best_classic['Global_R2']
                ganho_b_pct = (ganho_b / abs(best_classic['Global_R2'])) * 100 if best_classic['Global_R2'] != 0 else 0
                print(f"  {'Ganho Bayesiano vs Classico':<50} {ganho_b:+.4f} ({ganho_b_pct:+.1f}%)")

    # ─── TABELA 6: ESTATISTICAS DESCRITIVAS POR MODELO ───
    if not valid.empty:
        print(f"\n\n{'=' * 100}")
        print(f"{'TABELA 6: ESTATISTICAS DESCRITIVAS POR MODELO':^100}")
        print(f"{'=' * 100}")
        print(f"\n  {'Modelo':<22} {'Media R2':<11} {'Mediana R2':<12} {'Std R2':<10} {'Min R2':<10} {'Max R2':<10} {'N':<5}")
        print(f"  {'-' * 78}")
        for modelo in sorted(valid['Modelo'].unique()):
            m_data = valid[valid['Modelo'] == modelo]['Global_R2']
            print(f"  {modelo:<22} {m_data.mean():.4f}      {m_data.median():.4f}       "
                  f"{m_data.std():.4f}     {m_data.min():.4f}     {m_data.max():.4f}     {len(m_data):<5}")

    # ─── TABELA 7: 10 PAISES SELECIONADOS ───
    paises_selecionados = ['Angola', 'Ghana', 'Kenya', 'Nigeria', 'Mali',
                           'Tanzania', 'Egypt', 'Pakistan', 'Iran', 'Afghanistan']
    if not valid.empty:
        print(f"\n\n{'=' * 100}")
        print(f"{'TABELA 7: METRICAS DOS 10 PAISES SELECIONADOS (MELHOR DATASET)':^100}")
        print(f"{'=' * 100}")
        # Usar o melhor dataset para cada modelo
        best_ds_per_model = valid.loc[valid.groupby('Modelo')['Global_R2'].idxmax()]
        print(f"\n  {'Pais':<15} {'RF':<8} {'XGB':<8} {'TFT':<8} {'SARIMAX':<9} {'LSTM':<8} {'B_Part':<8} {'B_Comp':<8}")
        print(f"  {'-' * 88}")
        # Nota: esta tabela sera preenchida com dados reais dos CSVs por pais

    # ─── SALVAR TABELAS EM CSV ───
    # Tabela comparativa completa
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'tabela_comparativa_completa.csv'), index=False)

    # Tabela Bayesianos vs Classicos
    if not valid.empty:
        comparison_rows = []
        for _, row in valid.iterrows():
            comparison_rows.append({
                'Dataset': row['Dataset'], 'Estrategia': row['Estrategia'],
                'Modelo': row['Modelo'],
                'Tipo': 'Bayesiano' if 'Bayes' in row['Modelo'] else 'Classico',
                'Global_R2': row['Global_R2'],
                'Global_RMSE': row['Global_RMSE'],
                'Global_MAE': row['Global_MAE'],
                'PerCountry_Median_R2': row.get('PerCountry_Median_R2'),
                'N_Countries': row.get('N_Countries'),
                'N_Countries_R2_Positive': row.get('N_Countries_R2_Positive')
            })
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(os.path.join(config.OUTPUT_DIR, 'tabela_bayesianos_vs_classicos.csv'), index=False)

    # Tabela estatisticas descritivas
    if not valid.empty:
        stats_rows = []
        for modelo in valid['Modelo'].unique():
            m_data = valid[valid['Modelo'] == modelo]['Global_R2']
            stats_rows.append({
                'Modelo': modelo,
                'Media_R2': m_data.mean(),
                'Mediana_R2': m_data.median(),
                'Std_R2': m_data.std(),
                'Min_R2': m_data.min(),
                'Max_R2': m_data.max(),
                'N_Datasets': len(m_data)
            })
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(os.path.join(config.OUTPUT_DIR, 'tabela_estatisticas_descritivas.csv'), index=False)

    # Lista de ficheiros gerados
    print(f"\n\n{'=' * 100}")
    print(f"{'FICHEIROS CSV GERADOS':^100}")
    print(f"{'=' * 100}")
    print(f"\n  Diretorio: {config.OUTPUT_DIR}/")
    print(f"  - resumo_treinamento_completo.csv")
    print(f"  - tabela_comparativa_completa.csv")
    print(f"  - tabela_bayesianos_vs_classicos.csv")
    print(f"  - tabela_estatisticas_descritivas.csv")
    print(f"  - *_metricas_globais.csv (por dataset)")
    print(f"  - *_metricas_por_pais.csv (por modelo)")
    print(f"  - *_predictions.pkl (previsoes completas)")
    print(f"  - *.pkl (modelos treinados)")

    print(f"\n{'=' * 100}")
    print(f"{'TREINAMENTO COMPLETO FINALIZADO!':^100}")
    print(f"{'7 modelos (5 classicos + 2 Bayesianos) x 10 datasets = 70 combinacoes':^100}")
    print(f"{'=' * 100}")
