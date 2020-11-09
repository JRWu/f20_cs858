#!/bin/bash

cd /f20_cs858/models

MODEL=GoogleNews-vectors-negative300.bin

if test -f "$MODEL"; then
	echo "$MODEL exists, not downloading ..."
else
	echo "$MODEL does not exist, downloading and unzipping ..."
	wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
	gzip -d /f20_cs858/models/GoogleNews-vectors-negative300.bin.gz
	echo "DONE!"
fi

exit 0
