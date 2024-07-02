#!/bin/sh
gdown https://drive.google.com/drive/folders/14JG_w0V58iex62bVUHSBDYGBUECbDdx9 --folder
mv FashionIQ_Dataset/ fashionIQ_dataset
cd fashionIQ_dataset
tar -xvzf images.tar.gz
rm images.tar.gz

