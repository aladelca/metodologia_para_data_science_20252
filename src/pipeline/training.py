import os
import pickle  # nosec B403
import warnings
from typing import Any

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.preprocess import TimeSeriesPreprocessor
from src.utils import LSTMModel, auto_arima_manual

warnings.filterwarnings("ignore")


class TimeSeriesTrainer:
    """Comprehensive training class for multiple time series models."""

    def __init__(self, preprocessor=None):
        """
        Initialize trainer with preprocessor

        Args:
            preprocessor: TimeSeriesPreprocessor instance
        """
        self.preprocessor = preprocessor or TimeSeriesPreprocessor()
        self.models = {}
        self.training_results = {}

    def train_arima(
        self,
        data: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        param_p: int = 1,
        param_d: int = 2,
        param_q: int = 5,
        use_autorima: bool = False,
        criterio: str = "aic",
        save_path: str = None,
    ) -> tuple[Any, tuple[int, int, int], Any]:
        """
        Train ARIMA model with automatic parameter selection

        Args:
            data (pd.Series): Time series data
            max_p (int): Maximum p parameter
            max_d (int): Maximum d parameter
            max_q (int): Maximum q parameter
            criterio (str): Selection criteria ('aic' or 'bic')
            save_path (str): Path to save the model

        Returns:
            tuple: (fitted_model, best_order, results_df)
        """
        print("Training ARIMA model...")

        # Use auto ARIMA from utils
        if use_autorima:
            model, best_order, results_df = auto_arima_manual(
                data, max_p=max_p, max_d=max_d, max_q=max_q, criterio=criterio
            )
            self.training_results["arima"] = {
                "best_order": best_order,
                "aic": model.aic,
                "bic": model.bic,
                "results_df": results_df,
            }
        else:
            model = ARIMA(data, order=(param_p, param_d, param_q)).fit()
            results_df = None
            best_order = (param_p, param_d, param_q)
        # Store model
        self.models["arima"] = model

        # Save model if path provided
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"ARIMA model saved to {save_path}")

        print(f"ARIMA training completed. Best order: {best_order}")
        return model, best_order, results_df

    def train_sarimax(
        self,
        data,
        exog_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        save_path=None,
    ):
        """
        Train SARIMAX model with exogenous variables

        Args:
            data (pd.Series): Time series data
            exog_data (pd.DataFrame): Exogenous variables
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal order (P, D, Q, s)
            save_path (str): Path to save the model

        Returns:
            fitted SARIMAX model
        """
        print("Training SARIMAX model...")

        # Ensure data alignment
        aligned_data = data.loc[exog_data.index]
        aligned_exog = exog_data.loc[data.index]

        # Train SARIMAX model
        model = SARIMAX(
            aligned_data,
            exog=aligned_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(disp=False)

        # Store model
        self.models["sarimax"] = fitted_model
        self.training_results["sarimax"] = {
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
            "exog_columns": list(exog_data.columns),
        }

        # Save model if path provided
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(fitted_model, f)
            print(f"SARIMAX model saved to {save_path}")

        print("SARIMAX training completed")
        return fitted_model

    def train_prophet(self, df_train, exog_vars, save_path, **prophet_kwargs):
        """
        Train Prophet model

        Args:
            df_train (pd.DataFrame): Training data in Prophet format (ds, y)
            exog_vars (list): List of exogenous variable column names
            save_path (str): Path to save the model
            **prophet_kwargs: Additional Prophet parameters

        Returns:
            fitted Prophet model
        """
        print("Training Prophet model...")

        # Default Prophet parameters
        default_params = {
            "daily_seasonality": False,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.05,
        }
        default_params.update(prophet_kwargs)

        # Initialize Prophet model
        model = Prophet(**default_params)

        # Add regressors if provided
        if exog_vars:
            for regressor in exog_vars:
                if regressor in df_train.columns:
                    model.add_regressor(regressor)

        # Fit model
        model.fit(df_train)

        # Store model
        self.models["prophet"] = model
        self.training_results["prophet"] = {
            "exog_vars": exog_vars or [],
            "parameters": default_params,
            "components": (
                list(model.params.keys()) if hasattr(model, "params") else []
            ),
        }

        # Save model if path provided
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Prophet model saved to {save_path}")

        print("Prophet training completed")
        return model

    def train_catboost(
        self,
        train_features,
        train_target,
        test_features=None,
        test_target=None,
        save_path=None,
        **catboost_kwargs,
    ):
        """
        Train CatBoost model

        Args:
            train_features (pd.DataFrame): Training features
            train_target (pd.Series): Training target
            test_features (pd.DataFrame): Test features for validation
            test_target (pd.Series): Test target for validation
            save_path (str): Path to save the model
            **catboost_kwargs: Additional CatBoost parameters

        Returns:
            fitted CatBoost model
        """
        print("Training CatBoost model...")

        # Default parameters
        default_params = {
            "random_state": 123,
            "iterations": 1000,
            "verbose": False,
            "eval_metric": "RMSE",
        }
        default_params.update(catboost_kwargs)

        # Initialize model
        model = CatBoostRegressor(**default_params)

        # Prepare evaluation set if test data provided
        eval_set = None
        if test_features is not None and test_target is not None:
            eval_set = [(test_features, test_target)]

        # Train model
        model.fit(
            train_features,
            train_target,
            # eval_set=eval_set,
            use_best_model=bool(eval_set),
        )

        # Calculate feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": train_features.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Store model
        self.models["catboost"] = model
        self.training_results["catboost"] = {
            "parameters": default_params,
            "feature_importance": feature_importance,
            "best_iteration": getattr(
                model, "best_iteration_", default_params["iterations"]
            ),
        }

        # Save model if path provided
        if save_path:
            model.save_model(save_path)
            print(f"CatBoost model saved to {save_path}")

        print("CatBoost training completed")
        return model

    def train_lightgbm(
        self,
        train_features,
        train_target,
        test_features=None,
        test_target=None,
        save_path=None,
        **lgbm_kwargs,
    ):
        """
        Train LightGBM model

        Args:
            train_features (pd.DataFrame): Training features
            train_target (pd.Series): Training target
            test_features (pd.DataFrame): Test features for validation
            test_target (pd.Series): Test target for validation
            save_path (str): Path to save the model
            **lgbm_kwargs: Additional LightGBM parameters

        Returns:
            fitted LightGBM model
        """
        print("Training LightGBM model...")

        # Default parameters
        default_params = {
            "random_state": 123,
            "n_estimators": 1000,
            "verbose": -1,
            "metric": "rmse",
        }
        default_params.update(lgbm_kwargs)

        # Initialize model
        model = LGBMRegressor(**default_params)

        # Prepare evaluation set if test data provided
        eval_set = None
        if test_features is not None and test_target is not None:
            eval_set = [
                (test_features.astype(float), test_target.astype(float))
            ]

        # Train model
        model.fit(
            train_features.astype(float),
            train_target.astype(float),
            eval_set=eval_set,
        )

        # Calculate feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": train_features.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Store model
        self.models["lightgbm"] = model
        self.training_results["lightgbm"] = {
            "parameters": default_params,
            "feature_importance": feature_importance,
            "best_iteration": getattr(
                model, "best_iteration_", default_params["n_estimators"]
            ),
        }

        # Save model if path provided
        if save_path:
            joblib.dump(model, save_path)
            print(f"LightGBM model saved to {save_path}")

        print("LightGBM training completed")
        return model

    def train_lstm(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        input_size=None,
        hidden_size=50,
        num_layers=2,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=10,
        save_path=None,
        **lstm_kwargs,
    ):
        """
        Train LSTM model

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features for validation
            y_test (np.array): Test target for validation
            input_size (int): Input feature size
            hidden_size (int): Hidden layer size
            num_layers (int): Number of LSTM layers
            num_epochs (int): Training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            patience (int): Early stopping patience
            save_path (str): Path to save the model
            **lstm_kwargs: Additional LSTM parameters

        Returns:
            tuple: (trained_model, training_losses, validation_losses)
        """
        print("Training LSTM model...")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Determine input size
        if input_size is None:
            input_size = X_train.shape[2]

        # Initialize model
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            **lstm_kwargs,
        ).to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        test_loader = None
        if X_test is not None and y_test is not None:
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), torch.FloatTensor(y_test)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            if test_loader:
                model.eval()
                epoch_val_loss = 0

                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        epoch_val_loss += loss.item()

                avg_val_loss = epoch_val_loss / len(test_loader)
                val_losses.append(avg_val_loss)

                # Learning rate scheduling
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {avg_train_loss: .4f}, "
                        f"Val Loss: {avg_val_loss: .4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {avg_train_loss: .4f}"
                    )

        # Store model
        self.models["lstm"] = model
        self.training_results["lstm"] = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_epochs": epoch + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss if test_loader else None,
        }

        # Save model if path provided
        if save_path:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "output_size": 1,
                    },
                    "training_results": self.training_results["lstm"],
                },
                save_path,
            )
            print(f"LSTM model saved to {save_path}")

        print("LSTM training completed")
        return model, train_losses, val_losses

    def train_all_models(self, data_path, save_dir="models", **kwargs):
        """
        Train all models on the given dataset

        Args:
            data_path (str): Path to the data file
            save_dir (str): Directory to save models
            **kwargs: Additional parameters for specific models

        Returns:
            dict: Dictionary of trained models
        """
        print("Training all models...")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Load and preprocess data for different models
        print("\n=== Loading and preprocessing data ===")

        # Load raw data
        data = self.preprocessor.load_data(data_path)

        # 1. ARIMA
        print("\n=== Training ARIMA ===")
        arima_data = self.preprocessor.prepare_arima_data(data)
        self.train_arima(
            arima_data,
            save_path=os.path.join(save_dir, "arima_model.pkl"),
            **kwargs.get("arima", {}),
        )

        # 2. Prophet
        print("\n=== Training Prophet ===")
        prophet_data, exog_vars = self.preprocessor.prepare_prophet_data(data)
        self.train_prophet(
            prophet_data,
            exog_vars=exog_vars,
            save_path=os.path.join(save_dir, "prophet_model.pkl"),
            **kwargs.get("prophet", {}),
        )

        # 3. Machine Learning models
        print("\n=== Preparing ML data ===")
        train_X, train_y, test_X, test_y = self.preprocessor.prepare_ml_data(
            data
        )

        # CatBoost
        print("\n=== Training CatBoost ===")
        self.train_catboost(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=os.path.join(save_dir, "catboost_model"),
            **kwargs.get("catboost", {}),
        )

        # LightGBM
        print("\n=== Training LightGBM ===")
        self.train_lightgbm(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=os.path.join(save_dir, "lightgbm_model.pkl"),
            **kwargs.get("lightgbm", {}),
        )

        # 4. LSTM
        print("\n=== Preparing LSTM data ===")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_lstm_data(
            data
        )

        print("\n=== Training LSTM ===")
        self.train_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            save_path=os.path.join(save_dir, "lstm_model.pth"),
            **kwargs.get("lstm", {}),
        )

        print("\n=== All models trained successfully ===")
        return self.models

    def get_training_summary(self):
        """
        Get summary of all training results

        Returns:
            dict: Training summary
        """
        summary = {}
        for model_name, results in self.training_results.items():
            summary[model_name] = {"trained": True, "key_metrics": {}}

            if model_name in ["arima", "sarimax"]:
                summary[model_name]["key_metrics"] = {
                    "AIC": results.get("aic"),
                    "BIC": results.get("bic"),
                }
            elif model_name == "lstm":
                summary[model_name]["key_metrics"] = {
                    "final_train_loss": results.get("final_train_loss"),
                    "final_val_loss": results.get("final_val_loss"),
                    "best_val_loss": results.get("best_val_loss"),
                }
            elif model_name in ["catboost", "lightgbm"]:
                summary[model_name]["key_metrics"] = {
                    "best_iteration": results.get("best_iteration"),
                    "top_features": (
                        results.get("feature_importance", pd.DataFrame())
                        .head(5)
                        .to_dict("records")
                    ),
                }

        return summary
