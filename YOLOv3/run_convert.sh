#!/usr/bin/env sh

function yolov3()
{
     python convert2pb.py \
         --conf=./conf/yolo_darknet53_voc.json \
         --model_dir=./data/ckpt/ \
         --output_pb=./data/ckpt/yolo_darknet53_voc.pb
}


function yolofly()
{
    python convert2pb.py \
        --conf=./conf/yolofly_voc.json \
        --model_dir=./data/ckpt/ \
        --output_pb=./data/ckpt/yolofly_voc.pb
}


while getopts 'df' opt; do
    case ${opt} in
        d)
            yolov3
            ;;
        f)
            yolofly
            ;;
        ?)
            echo "Usage: `basename $0` [options]"
    esac
done

wait
