#!/usr/bin/env sh

entry=predict.py

function predict()
{
    python ${entry} \
        -i ./data/voc/test/VOCdevkit/VOC2007/JPEGImages \
        -f ./data/voc/test/VOCdevkit/VOC2007/test.txt \
        -o ./data/voc/test/VOCdevkit/VOC2007/predict/ \
        -m ./data/ckpt/yolo_darknet53_voc.pb \
        -c ./conf/yolo_darknet53_voc.json \
        -t voc \
        ;

}


function evaluate()
{
    python util/voc_eval.py \
        -d ./data/voc/test/VOCdevkit/VOC2007/predict/ \
        -a ./data/voc/test/VOCdevkit/VOC2007/Annotations/ \
        -c ./data/voc/test/VOCdevkit/VOC2007/cachedir/ \
        -f ./data/voc/test/VOCdevkit/VOC2007/test.txt \
        ;
}


while getopts 'pe' opt; do
    case ${opt} in
        p)
            predict
            ;;
        e)
            evaluate
            ;;
        ?)
            echo "Usage: `basename $0` [options]"
    esac
done

wait

