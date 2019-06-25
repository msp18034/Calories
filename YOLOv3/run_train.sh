#!/usr/bin/env sh

script="yolo.tar.gz"
entry="train_yolo.py"


function package()
{
    if [ -f ${script} ]
    then
        rm -f ${script}
    fi

    tar czf ${script} ${entry} network.py util/ model/
    echo "tar package updated"
}


function darknet53_param()
{
    user_defined_parameters="--local_mode=True \
                           --local_dir=data/voc/tfrecord/ \
                           --checkpointDir=data/ckpt/ \
                           --summary_dir=log/ \
                           --init_weights=conf/init/darknet53.conv.74 \
                           --model_conf=conf/yolo_darknet53_voc.json \
                           --restore=False \
                           --init_model_dir=data/ckpt/best/ \
                            "
}




function yolofly_param()
{
    user_defined_parameters="--local_mode=True \
                           --local_dir=data/voc/tfrecord/ \
                           --checkpointDir=data/ckpt/ \
                           --summary_dir=log/ \
                           --init_weights=conf/init/tiny.weights \
                           --model_conf=conf/yolofly_voc.json \
                           --restore=False \
                           --init_model_dir=data/ckpt/best/ \
                            "
}


function local_train()
{
    darknet53_param
    # yolofly_param

    python ${entry} \
        --task_index=0 \
        --job_name=ps \
        --ps_hosts=127.0.0.1:2222 \
        --worker_hosts=127.0.0.1:2224 \
        ${user_defined_parameters} &

    python ${entry} \
        --task_index=0 \
        --job_name=worker \
        --ps_hosts=127.0.0.1:2222 \
        --worker_hosts=127.0.0.1:2224 \
        ${user_defined_parameters} &
}


function distributed_train()
{
    package
    echo "error: not implemented..."
    echo "depending on specific distributed tf platform api"
}


function tensor_board()
{
    echo "error: not implemented..."
    echo "depending on specific distributed tf platform api"
}


while getopts 'rlt' opt; do
    case ${opt} in
        r)
            distributed_train
            ;;
        l)
            local_train
            ;;
        t)
            tensor_board
            ;;
        ?)
            echo "Usage: `basename $0` [options]"
    esac
done

wait
