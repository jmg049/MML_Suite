#!/bin/bash
set -e
ENV=$1
EXP_OUT_PATH=$2
DATA_PATH=$3

mkdir -p $EXP_OUT_PATH
mkdir -p $DATA_PATH

export EXP_PATH=$EXP_OUT_PATH
export DATA=$DATA_PATH

pip install git+https://github.com/jmg049/Modalities.git
pip install git+https://github.com/jmg049/DataSets.git


echo "Downloading data (this might take a while)"
mm_dataset --dataset avmnist --download_dir "$DATA_PATH" --unzip --del_zip
sed -i "s|\.\/AVMNIST\/dataset|$DATA_PATH/avmnist|g" $DATA_PATH/avmnist/*subset*.csv


mm_dataset --dataset kinetics-sounds --download_dir "$DATA_PATH" --unzip --del_zip
mv $DATA_PATH/kinetics-sounds $DATA_PATH/kinetics_sounds
sed -i "s|\.\/Kinetics_Sounds\/dataset|$DATA_PATH/kinetics_sounds|g" $DATA_PATH/kinetics_sounds/*.csv

mm_dataset --dataset mmimdb --download_dir "$DATA_PATH" --unzip --del_zip

mm_dataset --dataset cmu-mosi --download_dir "$DATA_PATH" --unzip --del_zip
mv $DATA_PATH/MOSI $DATA_PATH/mosi
rm -r $DATA_PATH/MOSI
mv $DATA_PATH/mosi/aligned_50.pkl $DATA_PATH/mosi/aligned.pkl 
mv $DATA_PATH/mosi/unaligned_50.pkl $DATA_PATH/mosi/unaligned.pkl 


mm_dataset --dataset cmu-mosei --download_dir "$DATA_PATH" --unzip --del_zip
mv $DATA_PATH/cmu-mosei $DATA_PATH/mosei
rm -r $DATA_PATH/cmu-mosei*
mv $DATA_PATH/mosei/aligned_50-002.pkl $DATA_PATH/mosei/aligned.pkl 

mm_dataset --dataset msp-improv --download_dir "$DATA_PATH" --unzip --del_zip
mv $DATA_PATH/MSP_features/MSP-IMPROV_features_2021/ $DATA_PATH/msp_improv/
rm -rf $DATA_PATH/MSP_features

mm_dataset --dataset iemocap --download_dir "$DATA_PATH" --unzip --del_zip