#!/bin/bash
function usage() {
    echo "Usage: $0 <task> [<optional args>]"
    echo "   task: PYTHON_BASE or CPU_BASE or GPU_BASE or DEV or GAMECORE"
    echo ""
    echo "Usage $0 COMMON_BASE [<common_base_out>]"
    echo "Usage $0 CPU_BASE [<common_base_in>] [<base_name_out>]"
    echo "Usage $0 GPU_BASE [<common_base_in>] [<base_name_out>]"
    echo "Usage $0 DEV [<base_name_in>] [<dev_name_out>] [<target_code_in>]"
    echo "Usage $0 GAMECORE [<base_name_in>] [<gamecore_name_out>]"
    echo ""
    echo "Example:"
    echo "    sh $0 COMMON_BASE common_base_py38"
    echo "    sh $0 CPU_BASE common_base_py38 cpu_base"
    echo "    sh $0 GPU_BASE common_base_py38 gpu_base"
    echo "    sh $0 DEV cpu_base cpu_dev code1v1"
    echo "    sh $0 DEV gpu_base gpu_dev code1v1"
    echo "    sh $0 DEV cpu_base cpu_dev code3v3"
    echo "    sh $0 DEV gpu_base gpu_dev code3v3"
    echo "    sh $0 GAMECORE cpu_dev gamecore_dev"
    exit 1
}
if [ $# -lt 1 ]; then
    usage
    exit -1
fi

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-"1"}

task=${1^^}

case "$task" in
"COMMON_BASE")
    common_base_out=${2-"common_base_py38"}
    echo "Execute: $0 COMMON_BASE $common_base_out"

    set -x
    docker build -f ./dockerfile/dockerfile.base -t ${common_base_out} .
    set +x
    ;;

"CPU_BASE")
    common_base_in=${2-"common_base_py38"}
    base_name_out=${3-"cpu_base"}
    echo "Execute: $0 CPU_BASE $common_base_in $base_name_out"

    set -x
    # Note: tensorflow + pytroch in one image use the dockerfile.base.cpu (DEPRECATED)
    # docker build -f ./dockerfile/dockerfile.base.cpu -t ${base_name_out} --build-arg=BASE_IMAGE=${common_base_in} .
    docker build -f ./dockerfile/dockerfile.base.torch --target cpu -t ${base_name_out} --build-arg=BASE_IMAGE=${common_base_in} .
    set +x
    ;;

"GPU_BASE")
    common_base_in=${2-"common_base_py38"}
    base_name_out=${3-"gpu_base"}
    echo "Execute: $0 GPU_BASE $common_base_in $base_name_out"

    set -x
    # Note: tensorflow + pytroch in one image use the dockerfile.base.gpu (DEPRECATED)
    # docker build -f ./dockerfile/dockerfile.base.gpu -t ${build_horovod_out} --target build_horovod --build-arg=BASE_IMAGE=${common_base_in} --build-arg=BUILD_PYTHON_IMAGE=${build_python_in} .
    docker build -f ./dockerfile/dockerfile.base.torch --target gpu -t ${base_name_out} --build-arg=BASE_IMAGE=${common_base_in} .
    set +x

    ;;

"DEV")
    base_name_in=${2-"cpu_base"}
    dev_name_out=${3-"cpu_dev"}
    target_code_in=${4-"code1v1"}
    dev_base_layer=${5-"dev_base"}
    echo "Execute: $0 DEV $base_name_in $dev_name_out $target_code_in $dev_base_layer"

    set -x
    docker build -f ./dockerfile/dockerfile.dev -t ${dev_name_out} --target ${target_code_in} --build-arg=BASE_IMAGE=${base_name_in} --build-arg=DEV_BASE=${dev_base_layer} .
    set +x
    ;;

"GAMECORE")
    for file in $(grep COPY ./dockerfile/dockerfile.gamecore | grep -v "#" | grep -v "\-\-from" | awk '{print $2}'); do
        if [ -n "$file" ]; then
            if [ ! -e $file ]; then
                echo -e "\033[31m not exist: $file \033[0m"
                exit -3
            else
                size=$(du -sh $file | awk '{print $1;}')
                echo -e "\033[32m exist: $file $size \033[0m"
            fi
        fi
    done
    base_name_in=${2-"cpu_dev"}
    internal_image=${3-"wine_base"}
    gamecore_name_out=${4-"gamecore_dev"}
    echo "Execute: $0 GAMECORE $base_name_in $internal_image $gamecore_name_out"

    set -x
    docker build -f ./dockerfile/dockerfile.gamecore -t ${gamecore_name_out} --build-arg=BASE_IMAGE=${base_name_in} --build-arg=INTERNAL_IMAGE=${internal_image} .
    set +x
    ;;
*)
    echo "Unknown task: $1"
    usage
    echo -2
    ;;
esac
