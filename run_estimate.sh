
MODEL_DIR=ckpts/yolov5/yolov5x6.onnx
MDDEL_TYPE='onnx'
NUM_MEMORY=4
UNIT_MEMORY='GB'
PARAMS=500000000
python3 estimate.py -m $MODEL_DIR -t $MDDEL_TYPE -n $NUM_MEMORY -u $UNIT_MEMORY
#python3 estimate.py -p $PARAMS -n 4 -u 'GB'
