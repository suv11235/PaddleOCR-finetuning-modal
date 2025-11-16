#!/bin/bash
# Helper script to build the Modal image with proper timeout handling
# The PaddlePaddle wheel is 1.7GB and can take time to download

echo "Building Modal image in detached mode..."
echo "This will take 10-15 minutes due to large PaddlePaddle download (1.7GB)"
echo ""

# Build the image by running a simple function in detached mode
modal run --detach modal_train.py::download_sample_dataset

echo ""
echo "Build started in background. Check status at:"
echo "https://modal.com/apps"
echo ""
echo "Once complete, you can run:"
echo "  modal run modal_train.py --mode download-dataset"
echo "  modal run modal_train.py --mode train"
