#!/bin/bash

cd /f20_cs858/models

# Obtain the model
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

# Unzip the model and delete the zipped model.
gzip -d /f20_cs858/models/GoogleNews-vectors-negative300.bin.gz

