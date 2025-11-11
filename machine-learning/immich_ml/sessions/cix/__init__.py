from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from immich_ml.config import log
from immich_ml.schemas import SessionNode

try:
    from .utils.NOE_Engine import EngineInfer
except (ImportError, OSError) as exc:  # pragma: no cover - depends on platform runtime
    EngineInfer = None  # type: ignore[assignment]
    is_available = False
    _load_error: Exception | None = exc
else:
    is_available = True
    _load_error = None


class CixNode(NamedTuple):
    name: str | None
    shape: tuple[int, ...]


class CixSession:
    """
    Thin wrapper around the CIX NOE engine so it can be used as a drop-in replacement
    for ONNXRuntime sessions inside Immich.
    """

    def __init__(self, model_path: Path | str) -> None:
        if EngineInfer is None:
            detail = f": {_load_error}" if _load_error is not None else ""
            raise RuntimeError(f"libnoe (CIX runtime) is not available{detail}")

        self.model_path = Path(model_path)

        log.info("Loading CIX model %s ...", self.model_path)
        self.engine = EngineInfer(self.model_path.as_posix())
        log.info("Loaded CIX model %s", self.model_path)

        self._inputs: list[CixNode] = self._build_nodes(self.engine.in_tensor_desc)
        self._outputs: list[CixNode] = self._build_nodes(self.engine.out_tensor_desc)

        self._input_name_to_index: dict[str, int] = self._index_nodes(self._inputs)
        self._output_name_to_index: dict[str, int] = self._index_nodes(self._outputs)

    def __del__(self) -> None:
        try:
            self.engine.clean()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def get_inputs(self) -> list[SessionNode]:
        return list(self._inputs)

    def get_outputs(self) -> list[SessionNode]:
        return list(self._outputs)

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,  # noqa: ARG002 - kept for API parity
    ) -> list[NDArray[np.float32]]:
        inputs = self._prepare_inputs(input_feed)
        outputs = [np.asarray(out, dtype=np.float32) for out in self.engine.forward(inputs)]

        if output_names is None:
            return outputs

        selected_outputs: list[NDArray[np.float32]] = []
        for name in output_names:
            idx = self._output_name_to_index.get(name)
            if idx is None:
                raise KeyError(f"Unknown output tensor name '{name}' for model {self.model_path}")
            selected_outputs.append(outputs[idx])
        return selected_outputs

    def _prepare_inputs(
        self,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
    ) -> list[NDArray[np.float32] | NDArray[np.int32]]:
        if not isinstance(input_feed, dict):
            raise TypeError("input_feed must be a dictionary mapping input names to numpy arrays")

        prepared: list[NDArray[np.float32] | NDArray[np.int32]] = []
        used_keys: set[str] = set()
        feed_items: list[tuple[str, NDArray[np.float32] | NDArray[np.int32]]] = list(input_feed.items())

        for idx, node in enumerate(self._inputs):
            key: str | None = node.name
            array: NDArray[np.float32] | NDArray[np.int32] | None = None

            if key is not None and key in input_feed:
                array = input_feed[key]
                used_keys.add(key)
            elif len(self._inputs) == 1 and feed_items:
                if len(feed_items) != 1:
                    raise KeyError(
                        f"Expected a single input but received {len(feed_items)} for model {self.model_path}",
                    )
                # single-input models: accept whatever key the caller used
                key, array = feed_items[0]
                used_keys.add(key)
            else:
                # fall back to positional ordering if the shapes match and cardinality is correct
                if len(feed_items) != len(self._inputs):
                    missing = key if key is not None else f"index {idx}"
                    raise KeyError(f"Expected input '{missing}' for model {self.model_path}")
                key, array = feed_items[idx]
                if key in used_keys:
                    raise KeyError(f"Duplicate value detected for input '{key}'")
                used_keys.add(key)

            np_array = np.asarray(array)
            if not np_array.flags.c_contiguous:
                np_array = np.ascontiguousarray(np_array)
            prepared.append(np_array)

        return prepared

    def _build_nodes(self, descriptors: Iterable[Any]) -> list[CixNode]:
        return [CixNode(name=self._extract_name(descriptor), shape=self._extract_shape(descriptor)) for descriptor in descriptors]

    def _index_nodes(self, nodes: Iterable[CixNode]) -> dict[str, int]:
        index: dict[str, int] = {}
        for idx, node in enumerate(nodes):
            if node.name is not None:
                index[node.name] = idx
        return index

    def _extract_name(self, descriptor: Any) -> str | None:
        name = getattr(descriptor, "name", None)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        if isinstance(name, str):
            return name or None
        return None

    def _extract_shape(self, descriptor: Any) -> tuple[int, ...]:
        """
        Attempt to read the tensor shape from the NOE tensor descriptor.
        The descriptor exposes the shape as either an iterable `dims`,
        `shape`, or a sequence with an accompanying `dim_num`/`rank`. The
        implementation is defensive so that we can tolerate minor API
        differences across libnoe versions.
        """

        def _sanitize_dims(dims: Iterable[Any]) -> tuple[int, ...]:
            values: list[int] = []
            for dim in dims:
                try:
                    value = int(dim)
                except (TypeError, ValueError):
                    continue
                if value == 0 and values:
                    # many runtimes use zero-padding for unused dimensions
                    break
                values.append(value)
            return tuple(values)

        for attr in ("dims", "shape", "dim", "dimensions"):
            dims = getattr(descriptor, attr, None)
            if dims is None:
                continue
            try:
                sanitized = _sanitize_dims(dims)
            except TypeError:
                continue
            if sanitized:
                return sanitized

        dim_num = getattr(descriptor, "dim_num", getattr(descriptor, "rank", None))
        dims_attr = getattr(descriptor, "dims", None)
        if dim_num is not None and dims_attr is not None:
            try:
                dims_iterable = list(dims_attr)
            except TypeError:
                dims_iterable = []
            sanitized = tuple(int(dim) for dim in dims_iterable[: int(dim_num)] if int(dim) > 0)
            if sanitized:
                return sanitized

        size = getattr(descriptor, "size", None)
        if isinstance(size, int) and size > 0:
            return (size,)

        log.debug("Falling back to unknown shape for descriptor %r from %s", descriptor, self.model_path)
        return ()


__all__ = ["CixSession", "CixNode", "is_available"]

