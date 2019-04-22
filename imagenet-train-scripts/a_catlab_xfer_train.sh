#!/bin/bash
# call example:
# ./a_catlab_xfer_train.sh . /home/subravcr/Projects/lab-palmeri/data ~/Projects/lab-palmeri/trainedImagenet/myModels/xferLearning/net-epoch-52/baseModel-net-epoch-52.mat 
workDir=$1
dataDir=$2
xferTrainBaseNetToUse=$3
matlabCmd="matlab -nodisplay -nosplash"
trainFuncToUse='execTransferTraining'

#--------------------------------------------------
# Execute maltab script/function in a try-catch block
#--------------------------------------------------
function execTransferTraining(){
    if [ -d "$dataDir" ]; then
        echo "Executing matlab call"
        strCmd="$matlabCmd  -r \"addpath('$workDir');a_accre_xfer_train('$workDir','$dataDir','$xferTrainBaseNetToUse');exit;\""
        echo $strCmd
        eval $strCmd
    else
        echo "The imagenet dir ${dataDir} does not exist! Aborting training"
    fi
}
#--------------------------------------------------
# Call functions in order...
#--------------------------------------------------
$trainFuncToUse

