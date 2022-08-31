#!/bin/bash

THING="rgb_checkpoint.tar.gz"

echo "downloading rgb model from dropbox..."
wget https://www.dropbox.com/s/n93ryvrqyiram56/${THING}

echo "extracting from tar..."
tar -xvf ${THING}

echo "deleting the tar..."
rm -v ${THING}

echo "done"
