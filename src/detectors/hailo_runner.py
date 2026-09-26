"""HailoRT 推理封装。

Hailo-8 上同一时刻只有一个 VDevice, 不能多个进程/上下文各开一个。
所以本模块提供两种用法:
    - HailoRunner(单模型): 自己开 VDevice, 适用于只有一个模型的场景 (M1)
    - HailoMultiRunner(多模型): 共用一个 VDevice, 在 M2+ 同时跑多模型时使用

参照 reComputer-R20-CV 参考实现的模式,做一次单例化的 VDevice + 激活的
InferVStreams pipeline,避免每次推理重建 context 带来的开销。
"""
from __future__ import annotations

import logging
import time
from typing import Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HailoRunner:
    """单模型 HEF 推理包装 (独占 VDevice)。

    用法:
        >>> with HailoRunner("/path/to/model.hef") as runner:
        ...     out = runner.run({"input_layer1": x_uint8})

    注意: M2+ 同时需要多个模型时,不要用本类;改用 HailoMultiRunner。
    """

    def __init__(
        self,
        hef_path: str,
        input_format_uint8: bool = True,
        output_format_uint8: bool = False,
        device_interface: str = "PCIe",
        warmup_runs: int = 3,
    ):
        from hailo_platform import (  # noqa: F401
            HEF,
            VDevice,
            FormatType,
            HailoStreamInterface,
            ConfigureParams,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )
        self.hef_path = hef_path
        self.input_format = FormatType.UINT8 if input_format_uint8 else FormatType.FLOAT32
        self.output_format = FormatType.UINT8 if output_format_uint8 else FormatType.FLOAT32
        if device_interface == "PCIe":
            self.interface = HailoStreamInterface.PCIe
        else:
            raise ValueError(f"Unsupported interface: {device_interface}")

        self.warmup_runs = warmup_runs

        self._vdevice = None
        self._hef = None
        self._network_group = None
        self._ng_params = None
        self._infer_pipe = None
        self._pipe_ctx = None
        self._activation = None
        self._input_infos: list = []
        self._output_infos: list = []

    def __enter__(self) -> "HailoRunner":
        from hailo_platform import (
            VDevice, HEF, ConfigureParams,
            InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType,
        )
        self._vdevice = VDevice()
        self._hef = HEF(self.hef_path)
        cfg = ConfigureParams.create_from_hef(hef=self._hef, interface=self.interface)
        self._network_group = self._vdevice.configure(self._hef, cfg)[0]
        self._ng_params = self._network_group.create_params()
        self._input_infos = self._hef.get_input_vstream_infos()
        self._output_infos = self._hef.get_output_vstream_infos()
        in_p = InputVStreamParams.make(self._network_group, format_type=self.input_format)
        out_p = OutputVStreamParams.make(self._network_group, format_type=self.output_format)
        self._activation = self._network_group.activate(self._ng_params)
        self._activation.__enter__()
        self._pipe_ctx = InferVStreams(self._network_group, in_p, out_p)
        self._infer_pipe = self._pipe_ctx.__enter__()

        self._warmup()
        return self

    def _warmup(self) -> None:
        from hailo_platform import FormatType
        for i, info in enumerate(self._input_infos):
            shape = (1, *info.shape) if len(info.shape) >= 3 else (1, info.shape[0])
            if self.input_format == FormatType.UINT8:
                dummy = np.random.randint(0, 255, shape, dtype=np.uint8)
            else:
                dummy = np.random.uniform(-1, 1, shape).astype(np.float32)
            self._infer_pipe.infer({info.name: dummy})
        for _ in range(self.warmup_runs - 1):
            self._infer_pipe.infer({info.name: dummy})
        logger.info("HailoRunner warmed up: %s", self.hef_path)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._pipe_ctx is not None:
                self._pipe_ctx.__exit__(exc_type, exc, tb)
        except Exception:
            logger.exception("Pipe close failed")
        try:
            if self._activation is not None:
                self._activation.__exit__(exc_type, exc, tb)
        except Exception:
            logger.exception("Activation close failed")
        try:
            if self._vdevice is not None:
                self._vdevice.release()
        except Exception:
            logger.exception("VDevice release failed")

    @property
    def input_infos(self) -> list:
        return list(self._input_infos)

    @property
    def output_infos(self) -> list:
        return list(self._output_infos)

    def run(self, inputs: Mapping[str, np.ndarray]) -> dict:
        return self._infer_pipe.infer(inputs)

    def run_with_timing(self, inputs: Mapping[str, np.ndarray]) -> tuple[dict, float]:
        t0 = time.perf_counter()
        out = self._infer_pipe.infer(inputs)
        dt = (time.perf_counter() - t0) * 1000.0
        return out, dt


class _ModelSlot:
    """一个 HEF 在多模型 runner 里占一个 slot。"""

    def __init__(self, name: str, hef, network_group, in_fmt, out_fmt):
        self.name = name
        self.hef = hef
        self.network_group = network_group
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt
        self.activation = None
        self.pipe_ctx = None
        self.infer_pipe = None
        self.ng_params = None
        self.input_infos: list = []
        self.output_infos: list = []
        self.is_active = False
        self._warmed = False


class HailoMultiRunner:
    """多模型共享 VDevice 的 runner (含 context switching)。

    Hailo-8 单卡只能激活一个 NetworkGroup。多个 HEF 通过 configure() 同时装入,
    但 inference 时按需 swap activation。实测 swap 延迟 < 2ms。

    用法:
        >>> with HailoMultiRunner() as mr:
        ...     mr.add("yolov8_person", "/path/yolo.hef")
        ...     mr.add("scrfd_face", "/path/scrfd.hef")
        ...     mr.activate("yolov8_person")  # 预热
        ...     out = mr.run("yolov8_person", {"input": x})
        ...     mr.activate("scrfd_face")     # 按需切换
        ...     out = mr.run("scrfd_face", {"input": x})
        ...     mr.activate("yolov8_person")  # 切回
    """

    def __init__(self, device_interface: str = "PCIe", warmup_runs: int = 3):
        from hailo_platform import HailoStreamInterface  # noqa: F401
        self._vdevice = None
        self._slots: dict[str, _ModelSlot] = {}
        self._active_name: Optional[str] = None
        if device_interface == "PCIe":
            self.interface = HailoStreamInterface.PCIe
        else:
            raise ValueError(f"Unsupported interface: {device_interface}")
        self.warmup_runs = warmup_runs

    def __enter__(self) -> "HailoMultiRunner":
        from hailo_platform import VDevice
        self._vdevice = VDevice()
        logger.info("HailoMultiRunner: shared VDevice opened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        names = list(self._slots.keys())
        for name in names:
            self._remove_slot(name)
        try:
            if self._vdevice is not None:
                self._vdevice.release()
        except Exception:
            logger.exception("VDevice release failed")
        self._vdevice = None
        logger.info("HailoMultiRunner: VDevice released")

    def add(
        self,
        name: str,
        hef_path: str,
        input_format_uint8: bool = True,
        output_format_uint8: bool = False,
    ) -> _ModelSlot:
        from hailo_platform import (
            HEF, ConfigureParams,
            InputVStreamParams, OutputVStreamParams, FormatType,
        )
        if self._vdevice is None:
            raise RuntimeError("HailoMultiRunner not entered")
        if name in self._slots:
            raise ValueError(f"Model {name!r} already added")
        in_fmt = FormatType.UINT8 if input_format_uint8 else FormatType.FLOAT32
        out_fmt = FormatType.UINT8 if output_format_uint8 else FormatType.FLOAT32
        hef = HEF(hef_path)
        cfg = ConfigureParams.create_from_hef(hef=hef, interface=self.interface)
        ng = self._vdevice.configure(hef, cfg)[0]
        slot = _ModelSlot(
            name=name, hef=hef, network_group=ng,
            in_fmt=in_fmt, out_fmt=out_fmt,
        )
        slot.ng_params = ng.create_params()
        slot.input_infos = hef.get_input_vstream_infos()
        slot.output_infos = hef.get_output_vstream_infos()
        self._slots[name] = slot
        logger.info("HailoMultiRunner: configured model %r from %s", name, hef_path)
        return slot

    def activate(self, name: str) -> None:
        """切换激活到指定模型。如已激活则跳过。"""
        if name not in self._slots:
            raise KeyError(f"Model {name!r} not loaded")
        if self._active_name == name:
            return
        # Deactivate previous
        if self._active_name is not None:
            self._deactivate(self._active_name)
        # Activate target
        slot = self._slots[name]
        # 每次都新建 activation + pipe (因为 __exit__ 后不能复用)
        from hailo_platform import (
            InferVStreams, InputVStreamParams, OutputVStreamParams,
        )
        slot.activation = slot.network_group.activate(slot.ng_params)
        slot.activation.__enter__()
        in_p = InputVStreamParams.make(slot.network_group, format_type=slot.in_fmt)
        out_p = OutputVStreamParams.make(slot.network_group, format_type=slot.out_fmt)
        slot.pipe_ctx = InferVStreams(slot.network_group, in_p, out_p)
        slot.infer_pipe = slot.pipe_ctx.__enter__()
        slot.is_active = True
        if not slot._warmed:  # type: ignore[attr-defined]
            self._warmup(slot)
            slot._warmed = True  # type: ignore[attr-defined]
        self._active_name = name

    def _deactivate(self, name: str) -> None:
        slot = self._slots[name]
        if not slot.is_active or slot.activation is None:
            return
        try:
            slot.pipe_ctx.__exit__(None, None, None)
        except Exception:
            logger.exception("pipe_ctx exit failed for %s", name)
        try:
            slot.activation.__exit__(None, None, None)
        except Exception:
            logger.exception("activation exit failed for %s", name)
        slot.is_active = False

    def _warmup(self, slot: _ModelSlot) -> None:
        from hailo_platform import FormatType
        for info in slot.input_infos:
            shape = (1, *info.shape) if len(info.shape) >= 3 else (1, info.shape[0])
            if slot.in_fmt == FormatType.UINT8:
                dummy = np.random.randint(0, 255, shape, dtype=np.uint8)
            else:
                dummy = np.random.uniform(-1, 1, shape).astype(np.float32)
            slot.infer_pipe.infer({info.name: dummy})
        for _ in range(self.warmup_runs - 1):
            slot.infer_pipe.infer({info.name: dummy})

    def _remove_slot(self, name: str) -> None:
        slot = self._slots.pop(name, None)
        if slot is None:
            return
        # cleanup without recursive lookup
        if slot.is_active and slot.activation is not None:
            try:
                if slot.pipe_ctx is not None:
                    slot.pipe_ctx.__exit__(None, None, None)
            except Exception:
                logger.exception("pipe_ctx exit failed for %s", name)
            try:
                slot.activation.__exit__(None, None, None)
            except Exception:
                logger.exception("activation exit failed for %s", name)
            slot.is_active = False
        slot.activation = None
        slot.pipe_ctx = None
        slot.infer_pipe = None

    def run(self, model_name: str, inputs: Mapping[str, np.ndarray]) -> dict:
        if self._active_name != model_name:
            self.activate(model_name)
        slot = self._slots[model_name]
        return slot.infer_pipe.infer(inputs)

    def run_with_timing(self, model_name: str, inputs: Mapping[str, np.ndarray]) -> tuple[dict, float]:
        if self._active_name != model_name:
            self.activate(model_name)
        slot = self._slots[model_name]
        t0 = time.perf_counter()
        out = slot.infer_pipe.infer(inputs)
        dt = (time.perf_counter() - t0) * 1000.0
        return out, dt

    def get_slot(self, model_name: str) -> _ModelSlot:
        if model_name not in self._slots:
            raise KeyError(f"Model {model_name!r} not loaded")
        return self._slots[model_name]

    @property
    def active_model(self) -> Optional[str]:
        return self._active_name