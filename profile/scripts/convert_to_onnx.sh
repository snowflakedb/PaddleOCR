#!/bin/bash

DET_PATH=~/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/
REC_PATH=~/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/
CLS_PATH=~/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/
SAVE_DIR_PATH=~/models/paddle2onnx

paddle2onnx --model_dir $DET_PATH \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file $SAVE_DIR_PATH/det.onnx \
--opset_version 14 \
--enable_onnx_checker True \
--custom_ops '{"paddle_op":"onnx_op"}'

paddle2onnx --model_dir $REC_PATH \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file $SAVE_DIR_PATH/rec.onnx \
--opset_version 14 \
--enable_onnx_checker True \
--custom_ops '{"paddle_op":"onnx_op"}'

paddle2onnx --model_dir $CLS_PATH \
--model_filename ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel \
--params_filename ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams \
--save_file @SAVE_DIR_PATH/cls.onnx \
--opset_version 14 \
--enable_onnx_checker True \
--custom_ops '{"paddle_op":"onnx_op"}'
