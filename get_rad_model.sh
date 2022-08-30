#!/bin/bash

echo "downloading rad model from dropbox..."
wget https://www.dropbox.com/s/abnzjdp0eqpfpkr/rad_checkpoint.tar.gz

echo "extracting from tar..."
tar -xvf ${THING}.tar.gz

echo "deleting the tar..."
rm -v ${THING}.tar.gz

echo "done"
