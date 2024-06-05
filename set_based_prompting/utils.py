import datetime
import os
import typing
import pytz

base_tz = pytz.timezone("America/New_York")


def get_HF_TOKEN() -> typing.Union[str,None]:
    token = os.getenv("HF_TOKEN")
    # if token is None:
    #    raise ValueError("Please set the HF_TOKEN environment variable")
    return token


def get_timestamp() -> datetime.datetime:
    return datetime.datetime.now(base_tz)


def get_timestamp_str() -> str:
    return get_timestamp().strftime("%Y-%m-%d %H:%M:%S")


def short_timestamp_str() -> str:
    return get_timestamp().strftime("%H-%M")


def print_with_timestamp(s: str, *args, **kwargs) -> None:
    print(f"[{get_timestamp_str()}] {s}", *args, **kwargs)
