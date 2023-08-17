#!/bin/bash
function usage() {
    echo "Usage: $0 <task> [<optional args>]"
    echo "   task: PYTHON_BASE or CPU_BASE or GPU_BASE or DEV or GAMECORE"
    echo ""
    echo "Usage $0 COMMON_BASE [<PYTHON_VERSION>] [<common_base_out>] [<build_python_out>]"
    echo "Usage $0 CPU_BASE [<common_base_in>] [<base_name_out>]"
    echo "Usage $0 GPU_BASE [<common_base_in>] [<build_python_in>] [<base_name_out>] [<build_horovod_out>]"
    echo "Usage $0 DEV [<base_name_in>] [<dev_name_out>] [<target_code_in>]"
    echo "Usage $0 GAMECORE [<base_name_in>] [<gamecore_name_out>]"
    echo ""
    echo "Example:"
    echo "    $0 COMMON_BASE 3.7.16 common_base_py37"
    echo "    $0 COMMON_BASE 3.8.16 common_base_py38 build_python_py38"
    echo "    $0 CPU_BASE common_base_py37 cpu_base"
    echo "    $0 GPU_BASE common_base_py38 build_python_py38 gpu_base"
    echo "    $0 DEV cpu_base cpu_dev code1v1"
    echo "    $0 DEV gpu_base gpu_dev code1v1"
    echo "    $0 DEV cpu_base cpu_dev code3v3"
    echo "    $0 DEV gpu_base gpu_dev code3v3"
    echo "    $0 GAMECORE cpu_dev gamecore_dev"
    exit 1
}
if [ $# -lt 1 ]; then
    usage
    exit -1
fi

task=${1^^}

case "$task" in
"COMMON_BASE")
    PYTHON_VERSION_in=${2-"3.7.16"}
    common_base_out=${3-"common_base_py37"}
    build_python_out=${4-"build_python_py37"}
    echo "Execute: $0 COMMON_BASE $PYTHON_VERSION_in $common_base_out $build_python_out"

    set -x
    docker build -f ./dockerfile/dockerfile.base -t ${common_base_out} --build-arg=PYTHON_VERSION=${PYTHON_VERSION_in} .
    docker build -f ./dockerfile/dockerfile.base -t ${build_python_out} --build-arg=PYTHON_VERSION=${PYTHON_VERSION_in} --target build_python .
    set +x
    ;;

"CPU_BASE")
    common_base_in=${2-"common_base_py37"}
    base_name_out=${3-"cpu_base"}
    echo "Execute: $0 CPU_BASE $common_base_in $base_name_out"

    set -x
    docker build -f ./dockerfile/dockerfile.base.cpu -t ${base_name_out} --build-arg=BASE_IMAGE=${common_base_in} .
    set +x
    ;;

"GPU_BASE")
    common_base_in=${2-"common_base_py38"}
    build_python_in=${3-"build_python_py38"}
    base_name_out=${4-"gpu_base"}
    build_horovod_out=${5-"gpu_base_build_horovod"}
    echo "Execute: $0 GPU_BASE $common_base_in $build_python_in $base_name_out $build_python_out"

    set -x
    docker build -f ./dockerfile/dockerfile.base.gpu -t ${base_name_out} --build-arg=BASE_IMAGE=${common_base_in} --build-arg=BUILD_PYTHON_IMAGE=${build_python_in} .
    docker build -f ./dockerfile/dockerfile.base.gpu -t ${build_horovod_out} --target build_horovod --build-arg=BASE_IMAGE=${common_base_in} --build-arg=BUILD_PYTHON_IMAGE=${build_python_in} .
    set +x

    ;;

"DEV")
    base_name_in=${2-"cpu_base"}
    dev_name_out=${3-"cpu_dev"}
    target_code_in=${4-"code1v1"}
    echo "Execute: $0 DEV $base_name_in $dev_name_out $target_code_in"

    set -x
    docker build -f ./dockerfile/dockerfile.dev -t ${dev_name_out} --target ${target_code_in} --build-arg=BASE_IMAGE=${base_name_in} .
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
