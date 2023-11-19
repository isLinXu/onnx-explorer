

MODEL_DIR=ckpts/yolov5/yolov5x6.onnx
OUTPUT_DIR=weights/yolov5/yolov5x6
python3 main.py -m $MODEL_DIR -o $OUTPUT_DIR -s