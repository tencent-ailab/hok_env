#!/bin/bash

# usage: build_code.sh
# env: OUTPUT_DIR, default is ./build
# env: OUTPUT_FILENAME, default is code-$version.tgz

version=2.3.3-$(date +"%Y%m%d%H%M")
filename=code-$version.tgz

# current shell script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname $SCRIPT_DIR)
TMP_DIR="/tmp/train/"

output_dir=${OUTPUT_DIR:-"$SCRIPT_DIR/../build"}
mkdir -p $output_dir
output_dir=$(cd -- "$output_dir" &>/dev/null && pwd)
filename=${OUTPUT_FILENAME:-"code-$version.tgz"}

# reset build code dir
rm -rf $TMP_DIR && mkdir -p $TMP_DIR

# build code

# 在集群训练会给环境注入TASK_ID, 通过这个变量判断是否是集群训练
if [ -n "$TASK_ID" ]; then
    rsync -a --exclude="checkpoints_*" \
        --exclude="**/checkpoints" \
        --exclude="**/checkpoint" \
        --exclude="GameAiMgr_*.txt" \
        --exclude="log" \
        $ROOT_DIR/code $TMP_DIR
else
    # 集群训练不打包learner的init model以减小代码包大小
    rsync -a --exclude="checkpoints_*" \
        --exclude="**/checkpoints" \
        --exclude="**/checkpoint" \
        --exclude="GameAiMgr_*.txt" \
        --exclude="log" \
        --exclude="code/learner/model/init" \
        $ROOT_DIR/code $TMP_DIR
fi

cp -r $ROOT_DIR/scripts $TMP_DIR

# generate version
echo "$version" >$TMP_DIR/version

# 打包
cd $TMP_DIR && tar -czf $output_dir/$filename . && cd -
touch $output_dir/${filename}.done
echo $output_dir/${filename} done
