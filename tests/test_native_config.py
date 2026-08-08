from pathlib import Path

from minimaxh3_service.config import Settings
from minimaxh3_service.native import NativeH3Runner


def runner(settings: Settings) -> NativeH3Runner:
    return NativeH3Runner(settings, lambda value: Path(value))


def test_dynamic_vram_arguments_select_cuda_device_and_headroom() -> None:
    arguments = runner(
        Settings(device="cuda:1", vram_mode="dynamic", reserve_vram_gb=1.5)
    )._core_arguments()

    assert arguments == [
        "--cuda-device",
        "1",
        "--reserve-vram",
        "1.5",
        "--enable-dynamic-vram",
    ]


def test_cpu_device_does_not_add_conflicting_vram_arguments() -> None:
    arguments = runner(Settings(device="cpu", vram_mode="lowvram"))._core_arguments()

    assert "--cpu" in arguments
    assert "--lowvram" not in arguments
