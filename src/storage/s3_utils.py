"""Helper utilities for interacting with Amazon S3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

S3_SCHEME = "s3"


def get_s3_client() -> BaseClient:
    """Return a shared S3 client instance."""

    return boto3.client("s3")


def is_s3_uri(uri: str) -> bool:
    """Check whether a string references an S3 object."""

    return urlparse(uri).scheme.lower() == S3_SCHEME


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an S3 URI into bucket and key components."""

    parsed = urlparse(uri)
    if parsed.scheme != S3_SCHEME or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")

    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI must include an object key: {uri}")

    return parsed.netloc, key


def build_s3_uri(bucket: str, key: str) -> str:
    """Create an S3 URI string."""

    return urlunparse((S3_SCHEME, bucket, key, "", "", ""))


def ensure_trailing_slash(prefix: str) -> str:
    """Ensure a prefix ends with a trailing slash for folder-like paths."""

    if prefix.endswith("/"):
        return prefix
    return f"{prefix}/"


def iter_objects(bucket: str, prefix: str) -> Iterator[Dict[str, Any]]:
    """Yield objects under a bucket/prefix."""

    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents", [])


def latest_object_key(bucket: str, prefix: str) -> Optional[Any]:
    """Return the key for the most recently modified object under a prefix."""

    latest = None
    latest_time = None
    for obj in iter_objects(bucket, prefix):
        if latest_time is None or obj["LastModified"] > latest_time:
            latest_time = obj["LastModified"]
            latest = obj["Key"]
    return latest


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    """Upload a local file to S3."""

    client = get_s3_client()
    client.upload_file(str(local_path), bucket, key)


def upload_directory(
    local_dir: Path, bucket: str, prefix: str
) -> Sequence[str]:
    """Upload all files within a local directory to S3."""

    uploaded = []
    normalized_prefix = ensure_trailing_slash(prefix)
    for path in local_dir.rglob("*"):
        if path.is_file():
            relative = path.relative_to(local_dir).as_posix()
            key = f"{normalized_prefix}{relative}"
            upload_file(path, bucket, key)
            uploaded.append(build_s3_uri(bucket, key))
    return uploaded


def object_exists(bucket: str, key: str) -> bool:
    """Check whether an S3 object exists."""

    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:  # pragma: no cover - network dependent
        if exc.response["Error"].get("Code") == "404":
            return False
        raise


@dataclass(frozen=True)
class S3Path:
    """Convenience representation of an S3 object."""

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return build_s3_uri(self.bucket, self.key)

    @staticmethod
    def from_uri(uri: str) -> "S3Path":
        bucket, key = split_s3_uri(uri)
        return S3Path(bucket=bucket, key=key)
