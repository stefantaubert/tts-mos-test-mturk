#!/bin/bash
ROOT_DIR="/tmp/create-upload-csv"

# Update package list
sudo apt update

# Install Python 3.10
sudo apt install python3-pip python3.10 python3.10-dev python3.10-distutils python3.10-venv -y

# Empty "ROOT_DIR"
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR"

# Create virtual environment
# All tools will be installed there and can only be run if the venv is activated
python3.10 -m venv "$ROOT_DIR/.venv"

# activate environment
source "$ROOT_DIR/.venv/bin/activate"

# Install/update tools for installing software
python3.10 -m pip install \
  wheel==0.38.4 \
  setuptools==65.6.3

# Install/update tools
python3.10 -m pip install \
  text-selection==0.0.2 \
  txt-utils==0.0.1 \
  textgrid-tools==0.0.7

rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR"

# Create symbolic link of the data from all "algorithms" (.wav & .TextGrid file pairs)
# In this case the algorithms are "original", "model1", "model2" and "model3"
rm -rf "$ROOT_DIR/data"
mkdir -p "$ROOT_DIR/data"
cp -rs "/tmp/original-valset" "$ROOT_DIR/data/original"
cp -rs "/tmp/model1-synthesized-valset" "$ROOT_DIR/data/model1"
cp -rs "/tmp/model2-synthesized-valset" "$ROOT_DIR/data/model2"
cp -rs "/tmp/model3-synthesized-valset" "$ROOT_DIR/data/model3"

# Create upload folder
rm -rf "$ROOT_DIR/upload"
mkdir -p "$ROOT_DIR/upload"
rsync -amL --include='*.wav' -f 'hide,! */' $ROOT_DIR/data/ "$ROOT_DIR/upload"

# Export all audio paths
textgrid-tools-cli grids export-audio-paths \
  "$ROOT_DIR/data" \
  "$ROOT_DIR/audios.txt" \
  --log "$ROOT_DIR/logs/1.log"

# Export all marks
textgrid-tools-cli grids export-marks \
  "$ROOT_DIR/data" \
  "words" --sep " " \
  "$ROOT_DIR/words.txt" \
  --log "$ROOT_DIR/logs/2.log"

# Replace multiple whitespace induced by pauses in the grids
txt-utils-cli replace \
  "$ROOT_DIR/words.txt" \
  " {2,}" " " \
  --log "$ROOT_DIR/logs/3.log"

# Remove leading and trailing whitespace
txt-utils-cli replace-line \
  "$ROOT_DIR/words.txt" \
  "(^ | $)" "" \
  --log "$ROOT_DIR/logs/4.log"

cp \
  "$ROOT_DIR/audios.txt" \
  "$ROOT_DIR/algorithms.txt"
  
txt-utils-cli replace \
  "$ROOT_DIR/algorithms.txt" \
  "$ROOT_DIR/data/([a-z-]+)/.+" "\1" \
  --log "$ROOT_DIR/logs/10.log"

cp \
  "$ROOT_DIR/audios.txt" \
  "$ROOT_DIR/files.txt"
  
txt-utils-cli replace \
  "$ROOT_DIR/files.txt" \
  ".+/([A-Za-z0-9-]+).wav" "\1" \
  --log "$ROOT_DIR/logs/11.log"

text-selection-cli dataset create \
  "$ROOT_DIR/words.txt" \
  "base" \
  "$ROOT_DIR/dataset.pkl" \
  --log "$ROOT_DIR/logs/6.log"
  
text-selection-cli weights create-uniform \
  "$ROOT_DIR/dataset.pkl" \
  1 \
  "$ROOT_DIR/uniform.npy" \
  --log "$ROOT_DIR/logs/7.log"
  
rm -rf "$ROOT_DIR/sets"
mkdir -p "$ROOT_DIR/sets"

touch "$ROOT_DIR/sets/sel-audios.txt"
touch "$ROOT_DIR/sets/sel-words.txt"
touch "$ROOT_DIR/sets/sel-algorithms.txt"
touch "$ROOT_DIR/sets/sel-files.txt"

# Create 60 HITs with 8 audio files per HIT
for i in {1..60}
do
  text-selection-cli subsets sort-by-line-nr \
    "$ROOT_DIR/dataset.pkl" \
    "base" \
    --log "$ROOT_DIR/sets/$i/logs/1.log"
    
  text-selection-cli subsets sort-by-shuffle \
    "$ROOT_DIR/dataset.pkl" \
    "base" --seed "1$i" \
    --log "$ROOT_DIR/sets/$i/logs/2.log"
    
  text-selection-cli subsets filter-duplicates \
    "$ROOT_DIR/dataset.pkl" \
    "base" \
    "ignore" \
    "$ROOT_DIR/words.txt" \
    --log "$ROOT_DIR/sets/$i/logs/3.log"
    
  text-selection-cli subsets select-fifo \
    "$ROOT_DIR/dataset.pkl" \
    "base" \
    "selection-$i" \
    "$ROOT_DIR/uniform.npy" \
    8 \
    --log "$ROOT_DIR/sets/$i/logs/5.log"
    
  text-selection-cli subsets export \
    "$ROOT_DIR/dataset.pkl" \
    "selection-$i" \
    "$ROOT_DIR/audios.txt" \
    "$ROOT_DIR/sets/$i/audios.txt" \
    --log "$ROOT_DIR/sets/$i/logs/6.log"
    
  text-selection-cli subsets export \
    "$ROOT_DIR/dataset.pkl" \
    "selection-$i" \
    "$ROOT_DIR/words.txt" \
    "$ROOT_DIR/sets/$i/words.txt" \
    --log "$ROOT_DIR/sets/$i/logs/7.log"
    
  text-selection-cli subsets export \
    "$ROOT_DIR/dataset.pkl" \
    "selection-$i" \
    "$ROOT_DIR/algorithms.txt" \
    "$ROOT_DIR/sets/$i/algorithms.txt" \
    --log "$ROOT_DIR/sets/$i/logs/11.log"
    
  text-selection-cli subsets export \
    "$ROOT_DIR/dataset.pkl" \
    "selection-$i" \
    "$ROOT_DIR/files.txt" \
    "$ROOT_DIR/sets/$i/files.txt" \
    --log "$ROOT_DIR/sets/$i/logs/13.log"
    
  text-selection-cli subsets select-all \
    "$ROOT_DIR/dataset.pkl" \
    "ignore" \
    "base" \
    --log "$ROOT_DIR/sets/$i/logs/8.log"
    
  txt-utils-cli merge \
    "$ROOT_DIR/sets/sel-audios.txt" \
    "$ROOT_DIR/sets/$i/audios.txt" \
    "$ROOT_DIR/sets/sel-audios.txt" \
    --log "$ROOT_DIR/sets/$i/logs/9.log"
    
  txt-utils-cli merge \
    "$ROOT_DIR/sets/sel-words.txt" \
    "$ROOT_DIR/sets/$i/words.txt" \
    "$ROOT_DIR/sets/sel-words.txt" \
    --log "$ROOT_DIR/sets/$i/logs/10.log"
    
  txt-utils-cli merge \
    "$ROOT_DIR/sets/sel-algorithms.txt" \
    "$ROOT_DIR/sets/$i/algorithms.txt" \
    "$ROOT_DIR/sets/sel-algorithms.txt" \
    --log "$ROOT_DIR/sets/$i/logs/12.log"
    
  txt-utils-cli merge \
    "$ROOT_DIR/sets/sel-files.txt" \
    "$ROOT_DIR/sets/$i/files.txt" \
    "$ROOT_DIR/sets/sel-files.txt" \
    --log "$ROOT_DIR/sets/$i/logs/14.log"
done

echo "" > "$ROOT_DIR/sets/sel-audios-anonymized.txt"
n_files=$(find "$ROOT_DIR/data" -name "*.wav" | wc -l)
for ((i=1;  i<=$n_files; i++)); do
  if [ $i -eq $n_files ]
  then
    echo -n "$ROOT_DIR/data-anonymized/$i.wav" >> "$ROOT_DIR/sets/sel-audios-anonymized.txt"
  else
    echo "$ROOT_DIR/data-anonymized/$i.wav" >> "$ROOT_DIR/sets/sel-audios-anonymized.txt"
  fi
done

# Create anonymized audio files
rm -rf "$ROOT_DIR/data-anonymized"
mkdir -p "$ROOT_DIR/data-anonymized"
while read -u 3 -r line1 && read -u 4 -r line2; do
  if [ ! -z "$line1" ]
  then
    # echo "$line1" "$line2"
    ln -s "$line1" "$line2"
  fi
done 3< "$ROOT_DIR/sets/sel-audios.txt" 4< "$ROOT_DIR/sets/sel-audios-anonymized.txt"

# Create MTurk file
cp \
  "$ROOT_DIR/sets/sel-audios-anonymized.txt" \
  "$ROOT_DIR/sets/sel-audios-anonymized-url.txt"

# Insert upload base URL here
UPLOAD_URL="UPLOAD_URL"
txt-utils-cli replace \
  "$ROOT_DIR/sets/sel-audios-anonymized-url.txt" \
  "$ROOT_DIR/data-anonymized" "$UPLOAD_URL/download?path=" \
  --log "$ROOT_DIR/logs/9.log"

cp \
  "$ROOT_DIR/sets/sel-audios-anonymized-url.txt" \
  "$ROOT_DIR/sets/upload.csv"

echo "" >> "$ROOT_DIR/sets/upload.csv"

txt-utils-cli replace \
  "$ROOT_DIR/sets/upload.csv" \
  "(.+)\n(.+)\n(.+)\n(.+)\n(.+)\n(.+)\n(.+)\n(.+)\n" "\"\1\",\"\2\",\"\3\",\"\4\",\"\5\",\"\6\",\"\7\",\"\8\"\n" \
  --log "$ROOT_DIR/logs/8.log"
  
echo -e "audio_url_1,audio_url_2,audio_url_3,audio_url_4,audio_url_5,audio_url_6,audio_url_7,audio_url_8$(cat "$ROOT_DIR/sets/upload.csv")" > "$ROOT_DIR/sets/upload.csv"

# Create Ground Truth file
echo "audio_url,algorithm,file" > "$ROOT_DIR/sets/ground-truth.csv"
while read -u 3 -r audio_url && read -u 4 -r algorithm && read -u 5 -r audio_file; do
  if [ ! -z "$audio_url" ]
  then
    echo \"$audio_url\",\"$algorithm\",\"$audio_file\" >> "$ROOT_DIR/sets/ground-truth.csv"
  fi
done 3< "$ROOT_DIR/sets/sel-audios-anonymized-url.txt" 4< "$ROOT_DIR/sets/sel-algorithms.txt" 5< "$ROOT_DIR/sets/sel-files.txt"
