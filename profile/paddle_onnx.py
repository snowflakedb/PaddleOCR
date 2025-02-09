from paddleocr import PaddleOCR

import os
import numpy as np
import timeit
import click

from profile.commons import load_pdf_bytes, pdf_to_image
import onnxruntime as ort


def _init_model(base_dir_path: str) -> PaddleOCR:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 8
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return PaddleOCR(
        use_onnx=True,
        det_model_dir=os.path.join(base_dir_path, "det.onnx"),
        rec_model_dir=os.path.join(base_dir_path, "rec.onnx"),
        cls_model_dir=os.path.join(base_dir_path, "cls.onnx"),
        ocr_version="PP-OCRv4",
        rec_batch_num=6,
        onnx_providers=["CPUExecutionProvider"],
        onnx_sess_options=sess_options,
    )


@click.command()
@click.option("--paddle-dir-path", default="/home/play/models/paddle2onnx/")
def main(
    paddle_dir_path: str,
):
    pdf_bytes = load_pdf_bytes()
    image = pdf_to_image(pdf_bytes, dpi=200)

    model = _init_model(paddle_dir_path)

    times = timeit.Timer(
        lambda: model.ocr(
            np.asarray(image),
            det=True,
            rec=True,
            cls=False,
        )
    ).repeat(repeat=4, number=1)

    print(f"Mean time: {np.mean(times[1:])}")
    print(f"Std time: {np.std(times[1:])}")


if __name__ == "__main__":
    main()
