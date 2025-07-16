#!/bin/bash

# Change to scannetpp-toolkit directory
cd scannetpp-toolkit

# Run iphone prep
python -m iphone.prepare_iphone_data ../preprocess/scannetpp/prepare_iphone_data.yml

# Run rendering
python -m common.render ../preprocess/scannetpp/render.yml

# Prepare training data
python -m semantic.prep.prepare_training_data ../preprocess/scannetpp/prepare_training_data.yml

# Prepare semantic ground truth
python -m semantic.prep.prepare_semantic_gt ../preprocess/scannetpp/prepare_semantic_gt.yml