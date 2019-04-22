#!/bin/bash
# Arg1:
#     Pass env var: $SLURM_JOBID as argument1
# Arg2:
#     Pass 1/0: 1 for test, 0 for real
#                1 = use ramdisk-imagenet12-test.tar.gz
#                0 = use ramdisk-imagenet12.tar.gz
# Arg3: Optional
#     Pass the full-path-to-traind-model-to-use for transfer learning/fine-tuning
#
# Example:
#     a_accre_train.sh $SLURM_JOBID 1
#     a_accre_train.sh $SLURM_JOBID 1 [result-full-path]/[base-net-to-use.mat] 
echo "Executing a_accre_train.sh"

jobId=$1
test=$2
if [ "$test" = "1" ]; then
    BASE_DIR=""
    tarFile="ramdisk-imagenet12-test.tar.gz"
else
    BASE_DIR=""
    tarFile="ramdisk-imagenet12.tar.gz"
fi

echo "jobId: "$jobId
echo " test: "$test

# For transfer learning / fine-tuning
if [ -z "$3" ]; then
    echo "Normal training (not xfer-training)"
    trainFuncToUse="execTraining"
else
    echo "Transfer/Xfer-training "
    xferTrainBaseNetToUse=$3
    echo "xferTrainBaseNetToUse:"$xferTrainBaseNetToUse
    trainFuncToUse="execTransferTraining"
fi
echo "trainFuncToUse: "$trainFuncToUse

###########################################################################
tmpBaseDir="${BASE_DIR}/tmp/${USER}"
tmpDir="${tmpBaseDir}-${jobId}"
dataDir="${tmpDir}/ramdisk"
workDir="${HOME}/Projects/lab-palmeri/imagenet-train-scripts"
resultDir="${BASE_DIR}/scratch/${USER}/trainedImagenet"
matlabCmd="matlab -nodisplay -nosplash"
echo "${tmpBaseDir}...."
#--------------------------------------------------
# Copy and extract Imagenet
#--------------------------------------------------
function prepareImagenet(){
    #echo "Extracting imagenet tarball to SSD"
    tarDir="${BASE_DIR}/scratch/${USER}/ImageNet12"
    echo "Creating ${tmpDir}"
    mkdir -p "$tmpDir"
    # Do not copy just extract
    #echo "Coping ${tarDir}/${tarFile} to ${tmpDir}/${tarFile}"
    #cp ${tarDir}/${tarFile} ${tmpDir}/${tarFile}
    echo "Extracting ${tarDir}/${tarFile} to ${tmpDir}"
    tar xzf ${tarDir}/${tarFile} -C ${tmpDir}
    echo "Done extracting ${tarDir}/${tarFile}"
}
#--------------------------------------------------
# Execute maltab script/function in a try-catch block
#--------------------------------------------------
function execTraining(){
    if [ -d "$dataDir" ]; then
        mkdir -p "${resultDir}"
        echo "Executing matlab call"
        strCmd="$matlabCmd  -r \"addpath('$workDir');a_accre_train('$workDir','$dataDir','$resultDir');exit;\""
        echo $strCmd
        eval $strCmd
    else
        echo "The imagenet dir ${dataDir} does not exist! Aborting training"
    fi
}

#--------------------------------------------------
# Execute maltab script/function in a try-catch block
#--------------------------------------------------
function execTransferTraining(){
    if [ -d "$dataDir" ]; then
        mkdir -p "${resultDir}"
        echo "Executing matlab call"
        strCmd="$matlabCmd  -r \"addpath('$workDir');a_accre_xfer_train('$workDir','$dataDir','$xferTrainBaseNetToUse');exit;\""
        echo $strCmd
        eval $strCmd
    else
        echo "The imagenet dir ${dataDir} does not exist! Aborting training"
    fi
}
#--------------------------------------------------
# Clean up tmp dir
#--------------------------------------------------
function cleanupImagenet(){
    if [ -d "$tmpDir" ]; then
        echo "Removing ${tmpDir}"
        rm -rf "${tmpDir}"
        echo "Done removing ${tmpDir}"
    else
        echo "Nothing to clean up or ${tmpDir} does not exist!"
    fi
}

#--------------------------------------------------
# Clean up existing tmp/USER* dirs
#--------------------------------------------------
function cleanupIfExist(){
   rm -rf $tmpBaseDir*
}

#--------------------------------------------------
# Call functions in order...
#--------------------------------------------------

cleanupIfExist
prepareImagenet
$trainFuncToUse
cleanupImagenet

