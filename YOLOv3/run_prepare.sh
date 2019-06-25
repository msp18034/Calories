#!/usr/bin/env bash
set -e

function gen_anchors()
{
     python util/gen_anchors.py \
         -c ./conf/yolo_darknet53_voc.json \
         -a ./data/voc/train/VOCdevkit/VOC2007/Annotations/ \
         -f ./data/voc/train/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
         -i ./data/voc/train/VOCdevkit/VOC2007/JPEGImages/
}

function gen_tfrecord()
{
     echo "begin to make tf-record for trainval ..."
     python util/make_tfrecord_parallel.py \
         --anno_folder='./data/voc/train/VOCdevkit/VOC2007/Annotations/' \
         --image_folder='./data/voc/train/VOCdevkit/VOC2007/JPEGImages/' \
         --name_file='./data/voc/train/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt' \
         --output_path='./data/voc/tfrecord/trainval/trainval.tfrecord' \
         --norm_width=500 \
         --norm_height=500

    echo "begin to make tf-record for test ..."
    python util/make_tfrecord_parallel.py \
        --anno_folder='./data/voc/test/VOCdevkit/VOC2007/Annotations/' \
        --image_folder='./data/voc/test/VOCdevkit/VOC2007/JPEGImages/' \
        --name_file='./data/voc/test/VOCdevkit/VOC2007/test.txt' \
        --output_path='./data/voc/tfrecord/test/test.tfrecord' \
        --norm_width=500 \
        --norm_height=500

}

while getopts 'at' OPT; do
    case $OPT in
        a)
            gen_anchors
            ;;
        t)
            gen_tfrecord
            ;;
        ?)
            echo "Usage: `basename $0` [options]"
    esac
done

wait
