#!/bin/bash

THING="rgb_checkpoint"

echo "working on ${THING}"

echo "downloading rgb model from dropbox..."
wget https://www.dropbox.com/s/n93ryvrqyiram56/${THING}.tar.gz

echo "extracting from tar..."
tar -xvf ${THING}.tar.gz

echo "deleting the tar..."
rm -v ${THING}.tar.gz

echo "done"
