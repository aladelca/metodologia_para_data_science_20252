"""AWS Glue job to extract stock data from Yahoo Finance and write it to S3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import yfinance as yf


def _configure_paths() -> None:
    """Ensure the `src` package is importable when the script runs in Glue."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.append(str(src_path))


_configure_paths()

# from preprocessing.extract import extract_stock_data  # noqa: E402


def extract_stock_data(ticker: str, period: str) -> Any:
    data = yf.Ticker(ticker)
    all_data = data.history(period=period)
    return all_data


LOGGER = logging.getLogger(__name__)
DEFAULT_BUCKET = os.environ.get("RAW_DATA_BUCKET", "raw-data-stocks")


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Download Yahoo Finance stock history and upload to S3",
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol to download",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="History period to request (e.g. '1y', 'max')",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=(
            "Destination S3 bucket (defaults to RAW_DATA_BUCKET env or "
            "'raw-data')"
        ),
    )
    parser.add_argument(
        "--object-key",
        dest="object_key",
        default=None,
        help=(
            "Object key inside the bucket. Generated automatically when "
            "omitted."
        ),
    )
    return parser.parse_known_args(argv)


def build_object_key(ticker: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"stock_data/{ticker.lower()}_{timestamp}.csv"


def upload_dataframe_to_s3(
    df: pd.DataFrame, bucket: str, object_key: str
) -> None:
    client = boto3.client("s3")
    buffer = BytesIO()
    df.reset_index().to_csv(buffer, index=False)
    buffer.seek(0)
    client.upload_fileobj(buffer, bucket, object_key)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args, unknown = parse_args(argv or sys.argv[1:])

    if unknown:
        LOGGER.info("Ignoring unknown Glue arguments: %s", unknown)

    LOGGER.info(
        "Downloading ticker %s with period %s", args.ticker, args.period
    )
    data = extract_stock_data(args.ticker, args.period)
    if data.empty:
        LOGGER.warning(
            "No data retrieved for ticker %s with period %s",
            args.ticker,
            args.period,
        )
        return

    object_key = args.object_key or build_object_key(args.ticker)
    LOGGER.info(
        "Uploading %s rows to s3://%s/%s",
        len(data),
        args.bucket,
        object_key,
    )
    upload_dataframe_to_s3(data, args.bucket, object_key)
    LOGGER.info("Upload completed")


if __name__ == "__main__":
    main()
