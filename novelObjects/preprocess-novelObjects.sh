#!/bin/bash
# file: preprocess-imagenet.sh

# Use as:
#   preprocess-imagenet.sh SRC_PATH DEST_PATH RGB/GRAY
#

data_dir=$1
out_dir=$2
colorspace=$3

# graphics magick (much faster)
num_cpus=1
method=gm

# image size
size=256 # most common

mkdir -p "$out_dir";
files=$(find $data_dir/*.tif -type f)
for f in $files
do
    of="$(basename $f)"
    ofile="$out_dir/${of//tif/jpg}"
    echo "Converting $f -> $ofile"OMP_NUM
    ${method} convert "TIFF:${f}" \
                 -verbose \
                 -quality 90 \
                 -colorspace "${colorspace}" \
                 -resize "${size}x${size}^" \
                 "JPEG:${ofile}"
done