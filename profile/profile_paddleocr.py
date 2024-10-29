from paddleocr import PaddleOCR

from PIL.Image import Image
import os
import numpy as np
from pdf2image import convert_from_bytes
import timeit
import cProfile
import click
from pydantic_settings import BaseSettings, SettingsConfigDict


class PaddleSettings(BaseSettings):
    use_gpu: bool = False
    use_xpu: bool = False
    use_mlu: bool = False
    use_angle_cls: bool = True
    enable_mkldnn: bool = True
    det: bool = True
    rec: bool = True
    cls: bool = True
    ocr_version: str = "PP-OCRv4"
    rec_batch_num: int = 6
    cpu_threads: int = 10
    benchmark: bool = False

    model_config = SettingsConfigDict(env_prefix="PADDLE_")


def _load_pdf_bytes() -> bytes:
    dir_path = os.path.dirname(__file__)
    path = os.path.join(dir_path, "resources", "sample_pdf.pdf")
    with open(path, "rb") as fh:
        return fh.read()


def pdf_to_image(pdf_bytes: bytes, dpi: int) -> Image:
    images = convert_from_bytes(pdf_bytes, dpi=dpi, grayscale=False)
    return images[0]


class TimeitDpi:
    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.dpi_values = [50, 100, 150, 200, 250, 300]
        self.mean_exec_time = [0] * len(self.dpi_values)
        self.std_exec_time = [0] * len(self.dpi_values)

    def setup(self) -> None:
        self.pdf_bytes = _load_pdf_bytes()
        self.model = PaddleOCR(**self.settings.model_dump())

    def timeit(self) -> None:
        self.setup()

        for idx in range(len(self.dpi_values)):
            image = pdf_to_image(self.pdf_bytes, self.dpi_values[idx])
            times = timeit.Timer(
                lambda: self.model.ocr(
                    np.asarray(image),
                    det=self.settings.det,
                    rec=self.settings.rec,
                    cls=self.settings.cls,
                )
            ).repeat(repeat=4, number=1)

            # Drop first call due to warm-up of the model.
            self.mean_exec_time[idx] = np.mean(times[1:]).item()
            self.std_exec_time[idx] = np.std(times[1:]).item()


class CprofilePaddle:
    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.dpi_values = [50, 100, 150, 200, 250, 300]

    def setup(self) -> None:
        self.pdf_bytes = _load_pdf_bytes()
        self.model = PaddleOCR(**self.settings.model_dump())

    def profile(self) -> None:
        self.setup()

        for dpi in self.dpi_values:
            image = pdf_to_image(self.pdf_bytes, dpi)
            with cProfile.Profile() as pr:
                self.model.ocr(
                    np.asarray(image),
                    det=self.settings.det,
                    rec=self.settings.rec,
                    cls=self.settings.cls,
                )
            os.makedirs("cprofile", exist_ok=True)
            pr.dump_stats(f"cprofile/profile_paddle_{dpi}.prof")


def print_recognition_pred():
    pdf_bytes = _load_pdf_bytes()
    model = PaddleOCR(det=False, rec=True, cls=False)
    image = pdf_to_image(pdf_bytes, dpi=300)
    ocr_result = model.ocr(np.asarray(image), det=False, rec=True, cls=False)
    print(ocr_result)


@click.command()
@click.option(
    "--mode",
    default="timeit",
    help="Mode to run the script in.",
    choices=["timeit", "cprofile", "print_recognition_pred"],
)
def main(mode: str) -> None:
    if mode == "timeit":
        settings = PaddleSettings()
        timeit_dpi = TimeitDpi(settings)
        timeit_dpi.timeit()
        for idx in range(len(timeit_dpi.dpi_values)):
            print(
                f"DPI: {timeit_dpi.dpi_values[idx]} | "
                f"{timeit_dpi.mean_exec_time[idx]:.4f} +/- "
                f"{timeit_dpi.std_exec_time[idx]:.4f}"
            )
    elif mode == "cprofile":
        settings = PaddleSettings(det=True, rec=True, cls=False)
        cprofile_paddle = CprofilePaddle(settings)
        cprofile_paddle.profile()
    elif mode == "print_recognition_pred":
        print_recognition_pred()


if __name__ == "__main__":
    main()
