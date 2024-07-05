#!/bin/sh
gdown https://drive.google.com/drive/u/0/folders/1MNeUd0Isw9qbPrtIZ9cyGx2cE-z1Q20H --folder
cd fashionIQ_dataset
tar -xvzf images.tar.gz
rm images.tar.gz

