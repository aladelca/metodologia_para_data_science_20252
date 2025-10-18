import json
import os
import tempfile
from typing import Any, Dict, List, Optional, no_type_check

import boto3
import pandas as pd

from src.pipeline.inference import TimeSeriesInference
from src.preprocessing.preprocess import TimeSeriesPreprocessor
from src.storage.s3_utils import (
    S3Path,
    build_s3_uri,
    ensure_trailing_slash,
    is_s3_uri,
    iter_objects,
    latest_object_key,
)

# Defaults (overridable via Lambda environment variables)
PREDICT_MODEL_BUCKET = os.environ.get(
    "PREDICT_MODEL_BUCKET", "raw-data-stocks"
)
PREDICT_MODEL_PREFIX = ensure_trailing_slash(
    (os.environ.get("PREDICT_MODEL_PREFIX", "models").strip("/"))
)
DATA_BUCKET = os.environ.get("TRAINING_DATA_BUCKET", "raw-data-stocks")
DATA_PREFIX = ensure_trailing_slash(
    (os.environ.get("TRAINING_DATA_PREFIX", "stock_data").strip("/"))
)
DATA_TICKER = os.environ.get("TRAINING_DATA_TICKER", "spy500")


MODEL_SUFFIX = {
    "arima": "arima_model.pkl",
    "prophet": "prophet_model.pkl",
    "catboost": "catboost_model.cbm",
    "lightgbm": "lightgbm_model.pkl",
    "lstm": "lstm_model.pth",
}


def _resolve_data_source(data_path: Optional[str]) -> str:
    """Resolve the dataset location.

    If a concrete S3 object or local path is provided, return it.
    If an S3 prefix is provided, pick the latest object under that prefix.
    Otherwise, resolve using DATA_BUCKET/DATA_PREFIX and prefer the latest
    file matching the configured ticker prefix.
    """

    if data_path:
        if is_s3_uri(data_path):
            s3 = S3Path.from_uri(data_path)
            # If it's a prefix (trailing slash), find the latest object
            if data_path.endswith("/"):
                key = latest_object_key(s3.bucket, s3.key)
                if not key:
                    raise FileNotFoundError(
                        f"No objects found under {data_path}"
                    )
                return build_s3_uri(s3.bucket, key)
            return data_path

        # Local path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return data_path

    # Default: pick latest object under configured prefix, preferring ticker
    prefix = ensure_trailing_slash(DATA_PREFIX)
    bucket = DATA_BUCKET

    # Prefer a specific ticker, e.g., stock_data/spy500_*.parquet
    ticker_prefix = f"{prefix}{DATA_TICKER.lower()}_"
    latest_time = None
    latest_key = None
    for obj in iter_objects(bucket, prefix):
        key = obj.get("Key", "")
        if not key:
            continue
        lm = obj.get("LastModified")
        if key.startswith(ticker_prefix) and (
            latest_time is None or (lm and lm > latest_time)
        ):
            latest_time = lm
            latest_key = key

    if not latest_key:
        latest_key = latest_object_key(bucket, prefix)

    if not latest_key:
        raise FileNotFoundError(
            f"No data objects found under s3: //{bucket}/{prefix}"
        )

    return build_s3_uri(bucket, latest_key)


def _find_latest_model_key(
    bucket: str, prefix: str, wanted_suffix: str
) -> Optional[Any]:
    """Find the most recently modified model object matching suffix."""
    latest = None
    latest_time = None
    normalized_prefix = ensure_trailing_slash(prefix)
    for obj in iter_objects(bucket, normalized_prefix):
        key = obj.get("Key", "")
        if not key or not key.endswith(wanted_suffix):
            continue
        lm = obj.get("LastModified")
        if latest_time is None or (lm and lm > latest_time):
            latest_time = lm
            latest = key
    return latest


def _business_days_between(start_date: str, end_date: str) -> int:
    idx = pd.date_range(start=start_date, end=end_date, freq="B")
    return len(idx)


def _business_date_index(start_date: str, end_date: str) -> List[str]:
    idx = pd.date_range(start=start_date, end=end_date, freq="B")
    return [d.date().isoformat() for d in idx]


def _download_s3_object(bucket: str, key: str, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path


def _serialize_predictions(
    model_type: str, preds: Any, start_date: str, end_date: str
) -> Dict[str, Any]:
    """Normalize predictions to a JSON-serializable structure."""
    if model_type == "prophet" and isinstance(preds, pd.DataFrame):
        out = preds[["ds", "yhat"]].copy()
        if "yhat_lower" in preds.columns and "yhat_upper" in preds.columns:
            out["yhat_lower"] = preds["yhat_lower"]
            out["yhat_upper"] = preds["yhat_upper"]
        # Bound to requested window if necessary
        mask = (out["ds"] >= pd.to_datetime(start_date)) & (
            out["ds"] <= pd.to_datetime(end_date)
        )
        out = out.loc[mask]
        return {
            "dates": [d.date().isoformat() for d in out["ds"].tolist()],
            "yhat": out["yhat"].astype(float).tolist(),
            "yhat_lower": out.get("yhat_lower", pd.Series(dtype=float))
            .astype(float)
            .tolist()
            if "yhat_lower" in out
            else None,
            "yhat_upper": out.get("yhat_upper", pd.Series(dtype=float))
            .astype(float)
            .tolist()
            if "yhat_upper" in out
            else None,
        }

    # For numeric arrays/Series, attach business date index
    dates = _business_date_index(start_date, end_date)
    if isinstance(preds, (list, tuple)):
        values = list(map(float, preds))
    elif hasattr(preds, "tolist"):
        values = list(map(float, preds.tolist()))  # numpy array or Series
    else:
        values = []
    # Truncate/align to date length for safety
    if len(values) != len(dates):
        values = values[: len(dates)]
    return {"dates": dates, "values": values}


def handler(
    event: Dict[str, Any], context: Any
) -> Dict[str, Any]:  # noqa: D401
    """AWS Lambda handler for prediction.

    Expected event payload:
      {
        "model_type": "arima|prophet|catboost|lightgbm|lstm",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "data_path": "s3://.../file.parquet" | "s3://.../prefix/" (optional),
        "output_s3_uri": "s3://bucket/prefix/predictions.csv" (optional)
      }
    """

    try:
        model_type = str(event.get("model_type", "")).strip().lower()
        start_date = str(event.get("start_date", "")).strip()
        end_date = str(event.get("end_date", "")).strip()
        data_path = event.get("data_path")
        output_s3_uri = event.get("output_s3_uri")

        if model_type not in MODEL_SUFFIX:
            raise ValueError("Invalid model_type")
        # Validate dates
        _ = pd.to_datetime(start_date)
        _ = pd.to_datetime(end_date)
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        # Resolve data source and load data
        resolved_data_uri = _resolve_data_source(data_path)
        preprocessor = TimeSeriesPreprocessor()
        data = preprocessor.load_data(resolved_data_uri)

        # Resolve model artifact in S3
        suffix = MODEL_SUFFIX[model_type]
        model_key = _find_latest_model_key(
            PREDICT_MODEL_BUCKET, PREDICT_MODEL_PREFIX, suffix
        )
        path_final = f"s3: //{PREDICT_MODEL_BUCKET}/{PREDICT_MODEL_PREFIX}"
        if not model_key:
            raise FileNotFoundError(f"No '{suffix}' found under {path_final}")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_path = _download_s3_object(
                PREDICT_MODEL_BUCKET, model_key, tmpdir
            )

            inf = TimeSeriesInference(preprocessor=preprocessor)
            model = inf.load_model(model_type, local_model_path)

            steps = _business_days_between(start_date, end_date)
            if steps <= 0:
                raise ValueError(
                    "Requested window has no business days to predict"
                )

            preds: Any
            if model_type == "arima":
                preds = inf.predict_arima(
                    model, steps=steps, return_conf_int=False
                )
            elif model_type == "prophet":
                preds = inf.predict_prophet(model, periods=steps, freq="B")
            elif model_type in ("catboost", "lightgbm"):
                # Prepare features for the requested window
                _, _, test_X, test_y = preprocessor.prepare_ml_data(
                    data,
                    train_start="1900-01-01",
                    train_end=start_date,
                    test_start=start_date,
                )
                raw = inf.predict_ml_models(
                    model, test_X, model_type=model_type
                )
                # Align to requested end date
                if not test_X.empty:
                    test_idx = test_X.index
                    mask = (test_idx >= pd.to_datetime(start_date)) & (
                        test_idx <= pd.to_datetime(end_date)
                    )
                    raw = (
                        list(raw[mask])
                        if len(raw) == len(test_idx)
                        else list(raw)
                    )
                preds = raw[:steps]
            elif model_type == "lstm":
                # Use last available sequence to roll forward
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                ) = preprocessor.prepare_lstm_data(data)
                initial = X_test[-1] if len(X_test) > 0 else X_train[-1]
                preds = inf.multi_step_forecast_lstm(
                    model,
                    initial_sequence=initial,
                    steps=steps,
                    target_scaler=preprocessor.target_scaler,
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            payload = {
                "model_type": model_type,
                "start_date": start_date,
                "end_date": end_date,
                "steps": steps,
                "model_s3_path": build_s3_uri(PREDICT_MODEL_BUCKET, model_key),
                "data_source": resolved_data_uri,
                "predictions": _serialize_predictions(
                    model_type, preds, start_date, end_date
                ),
            }

            # Optionally write predictions to S3 as CSV
            if output_s3_uri:
                s3p = S3Path.from_uri(output_s3_uri)
                df = _preds_to_dataframe(payload["predictions"], model_type)
                out_path = os.path.join(tmpdir, "predictions.csv")
                df.to_csv(out_path, index=False)
                boto3.client("s3").upload_file(out_path, s3p.bucket, s3p.key)
                payload["predictions_s3_uri"] = output_s3_uri

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(payload),
            }

    except Exception as exc:  # surface clear error for Lambda logs
        err = {"error": str(exc)}
        return {"statusCode": 500, "body": json.dumps(err)}


@no_type_check
def _preds_to_dataframe(preds, model_type: str) -> pd.DataFrame:
    """Convert predictions payload to a DataFrame (no strict typing)."""
    if model_type == "prophet":
        data = {
            "date": preds.get("dates", []),
            "yhat": preds.get("yhat", []),
        }
        if preds.get("yhat_lower") is not None:
            data["yhat_lower"] = preds["yhat_lower"]
        if preds.get("yhat_upper") is not None:
            data["yhat_upper"] = preds["yhat_upper"]
        return pd.DataFrame(data)
    return pd.DataFrame(
        {"date": preds.get("dates", []), "value": preds.get("values", [])}
    )
