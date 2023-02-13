import certifi
import json
import urllib3
import inspect
from enum import Enum
from typing import Optional, Any, Dict
from datetime import datetime
import pkg_resources  # part of setuptools
from .models.request import RequestOptionBuilder
from ..logging import get_logger
import logging
from ..exceptions import AuthError, BadResponse, NoResultsError

logger = get_logger("RESTClient")
version = "unknown"
try:
    version = pkg_resources.require("polygon-api-client")[0].version
except:
    pass


class BaseClient:
    def __init__(
        self,
        api_key: Optional[str],
        connect_timeout: float,
        read_timeout: float,
        num_pools: int,
        retries: int,
        base: str,
        verbose: bool,
        custom_json: Optional[Any] = None,
    ):
        if api_key is None:
            raise AuthError(
                f"Must specify env var POLYGON_API_KEY or pass api_key in constructor"
            )

        self.API_KEY = api_key
        self.BASE = base

        self.headers = {
            "Authorization": "Bearer " + self.API_KEY,
            "User-Agent": f"Polygon.io PythonClient/{version}",
        }

        # https://urllib3.readthedocs.io/en/stable/reference/urllib3.poolmanager.html
        # https://urllib3.readthedocs.io/en/stable/reference/urllib3.connectionpool.html#urllib3.HTTPConnectionPool
        self.client = urllib3.PoolManager(
            num_pools=num_pools,
            headers=self.headers,  # default headers sent with each request.
            ca_certs=certifi.where(),
            cert_reqs="CERT_REQUIRED",
        )

        self.timeout = urllib3.Timeout(connect=connect_timeout, read=read_timeout)
        self.retries = retries
        if verbose:
            logger.setLevel(logging.DEBUG)
        if custom_json:
            self.json = custom_json
        else:
            self.json = json

    def _decode(self, resp):
        return self.json.loads(resp.data.decode("utf-8"))

    def _get(
        self,
        path: str,
        params: Optional[dict] = None,
        result_key: Optional[str] = None,
        deserializer=None,
        raw: bool = False,
        options: Optional[RequestOptionBuilder] = None,
    ) -> Any:
        option = options if options is not None else RequestOptionBuilder()

        resp = self.client.request(
            "GET",
            self.BASE + path,
            fields=params,
            retries=self.retries,
            headers=self._concat_headers(option.headers),
        )

        if resp.status != 200:
            raise BadResponse(resp.data.decode("utf-8"))

        if raw:
            return resp

        obj = self._decode(resp)

        if result_key:
            if result_key not in obj:
                raise NoResultsError(
                    f'Expected key "{result_key}" in response {obj}.'
                    + "Make sure you have sufficient permissions and your request parameters are valid."
                    + f"This is the url that returned no results: {resp.geturl()}"
                )
            obj = obj[result_key]

        if deserializer:
            if type(obj) == list:
                obj = [deserializer(o) for o in obj]
            else:
                obj = deserializer(obj)

        return obj

    @staticmethod
    def time_mult(timestamp_res: str) -> int:
        if timestamp_res == "nanos":
            return 1000000000
        elif timestamp_res == "micros":
            return 1000000
        elif timestamp_res == "millis":
            return 1000

        return 1

    def _get_params(
        self, fn, caller_locals: Dict[str, Any], datetime_res: str = "nanos"
    ):
        params = caller_locals["params"]
        if params is None:
            params = {}
        # https://docs.python.org/3.8/library/inspect.html#inspect.Signature
        for argname, v in inspect.signature(fn).parameters.items():
            # https://docs.python.org/3.8/library/inspect.html#inspect.Parameter
            if argname in ["params", "raw"]:
                continue
            if v.default != v.empty:
                # timestamp_lt -> timestamp.lt
                val = caller_locals.get(argname, v.default)
                if isinstance(val, Enum):
                    val = val.value
                elif isinstance(val, bool):
                    val = str(val).lower()
                elif isinstance(val, datetime):
                    val = int(val.timestamp() * self.time_mult(datetime_res))
                if val is not None:
                    if (
                        argname.endswith("_lt")
                        or argname.endswith("_lte")
                        or argname.endswith("_gt")
                        or argname.endswith("_gte")
                        or argname.endswith("_any_of")
                    ):
                        argname = ".".join(argname.split("_", 1))
                    if argname.endswith("any_of"):
                        val = ",".join(val)
                    params[argname] = val

        return params

    def _concat_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        if headers is None:
            return self.headers
        return {**headers, **self.headers}

    def _paginate_iter(
        self,
        path: str,
        params: dict,
        deserializer,
        result_key: str = "results",
        options: Optional[RequestOptionBuilder] = None,
    ):
        while True:
            resp = self._get(
                path=path,
                params=params,
                deserializer=deserializer,
                result_key=result_key,
                raw=True,
                options=options,
            )
            decoded = self._decode(resp)
            for t in decoded[result_key]:
                yield deserializer(t)
            if "next_url" in decoded:
                path = decoded["next_url"].replace(self.BASE, "")
                params = {}
            else:
                return

    def _paginate(
        self,
        path: str,
        params: dict,
        raw: bool,
        deserializer,
        result_key: str = "results",
        options: Optional[RequestOptionBuilder] = None,
    ):
        if raw:
            return self._get(
                path=path,
                params=params,
                deserializer=deserializer,
                raw=True,
                options=options,
            )

        return self._paginate_iter(
            path=path,
            params=params,
            deserializer=deserializer,
            result_key=result_key,
            options=options,
        )
