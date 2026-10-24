#!/bin/bash

gdown https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM
cd mini-imagenet

for f in *.tar; do
    [ -e "$f" ] || continue
    tar -xf "$f"
    rm "$f"
done