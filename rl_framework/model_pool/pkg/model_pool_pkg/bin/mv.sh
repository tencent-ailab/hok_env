sourcePath=$2 #原始模型路径，是文件路径
destPath=$1 #批处理后的存储目录，是目录

tar -xf $sourcePath -C $destPath/
#mv $sourcePath $destPath/model
#cd $destPath && tar -xf model && rm -rf model

## checkpoint*
if [ `ls $destPath/checkpoints* | wc -l` -ne 0 ];then
   cd $destPath && mv checkpoints* checkpoint
fi

## footbool*
if [ `ls $destPath/epoch_*_complete | wc -l` -ne 0 ];then
   cd $destPath && mv epoch_*_complete checkpoint
fi

## save_model
if [ `ls $destPath/saved_model | wc -l` -ne 0 ];then
   cd $destPath && mv saved_model/1/model.savedmodel/ savedmodel && rm -rf saved_model
fi
