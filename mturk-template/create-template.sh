#!/usr/bin/env bash

# Set the amount of files per HIT
N_FILES_PER_HIT=8

BASEDIR=$(dirname "$0")
# header_path="$BASEDIR/header.html"
header_path="$BASEDIR/header-two-opts.html"
audio_path="$BASEDIR/audio.html"
footer_path="$BASEDIR/footer.html"

output_path="$BASEDIR/template.html"
cp "$header_path" "$output_path"

for (( i=1; i<=$N_FILES_PER_HIT; i++ ))
do
  txt-utils-cli merge \
    "$output_path" \
    "$audio_path" \
    "$output_path"
    
  txt-utils-cli replace \
    "$output_path" \
    "X" "$i"
done

txt-utils-cli merge \
  "$output_path" \
  "$footer_path" \
  "$output_path"
