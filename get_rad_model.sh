#!/bin/bash

THING="rad_checkpoint.tar.gz"

echo "downloading rad model from dropbox..."
wget https://www.dropbox.com/s/abnzjdp0eqpfpkr/${THING}

echo "extracting from tar..."
tar -xvf ${THING}

echo "deleting the tar..."
rm -v ${THING}

echo "done"
