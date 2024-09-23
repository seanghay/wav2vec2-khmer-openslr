#!/usr/bin/env bash

if [ ! -f "km_kh_male.zip" ]; then
  echo "downloading OpenSLR 42 dataset"
  gdown 1MDmQbwpVUiltHj85QEu6MtpVzbvMM-wM
fi

if [ ! -d "km_kh_male" ]; then
  unzip km_kh_male.zip
fi