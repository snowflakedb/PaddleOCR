from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
    quant_pre_process,
)
import os
import click


# TODO: produces error: Incomplete symbolic shape inference from
# onnxruntime.tools.symbolic_shape_infer line 2932.
def run(dir_path: str, filename: str):
    base_name = filename.split(".")[0]
    model_path = os.path.join(dir_path, filename)
    infer_model_path = os.path.join(dir_path, f"{base_name}_infer.onnx")
    quant_model_path = os.path.join(dir_path, f"{base_name}_quant.onnx")
    quant_pre_process(
        input_model=model_path,
        output_model_path=infer_model_path,
    )
    quantize_dynamic(
        model_input=infer_model_path,
        model_output=quant_model_path,
        weight_type=QuantType.QInt8,
    )


@click.command()
@click.option("--dir_path", type=str, required=True)
@click.option("--det_filename", type=str, required=True, default="det.onnx")
@click.option("--rec_filename", type=str, required=True, default="rec.onnx")
def main(dir_path: str, det_filename: str, rec_filename: str) -> None:
    for filename in [det_filename, rec_filename]:
        run(dir_path=dir_path, filename=filename)


if __name__ == "__main__":
    main()
