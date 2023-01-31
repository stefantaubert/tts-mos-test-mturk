# source /home/mi/code/tts-mos-test-mturk/create-template.sh

header_path="/home/mi/code/tts-mos-test-mturk/template/header.html"
audio_path="/home/mi/code/tts-mos-test-mturk/template/audio.html"
footer_path="/home/mi/code/tts-mos-test-mturk/template/footer.html"

output_path="/home/mi/code/tts-mos-test-mturk/template/template.html"
cp "$header_path" "$output_path"

for i in {1..8}
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
