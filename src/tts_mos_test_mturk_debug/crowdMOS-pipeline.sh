
mos-cli init \
  "/home/mi/code/tts-mos-test-mturk/examples/gen-gt.csv" \
  "/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv" \
  "/tmp/crowdMOS.pkl"

mos-cli ignore-too-fast \
  "/tmp/crowdMOS.pkl" \
  8 \
  "fast-workers"

mos-cli ignore-bad-workers \
  "/tmp/crowdMOS.pkl" \
  0.25 --mode "algorithm" \
  "bad-workers"
  

mos-cli ignore-by-listening-device \
  "/tmp/crowdMOS.pkl" \
  "laptop" "desktop" \
  "fast-workers & bad-workers -> no-headphones" \
  -m \
    "fast-workers" \
    "bad-workers"
  
# 0.7
mos-cli ignore-bad-workers \
  "/tmp/crowdMOS.pkl" \
  0.3 \
  "fast-workers & bad-workers -> no-headphones -> outliers-1" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones"

mos-cli ignore-too-few \
  "/tmp/crowdMOS.pkl" \
  30 \
  "fast-workers & bad-workers -> no-headphones -> outliers-2" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones"

mos-cli stats \
  "/tmp/crowdMOS.pkl" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones" \
    "fast-workers & bad-workers -> no-headphones -> outliers-1" \
    "fast-workers & bad-workers -> no-headphones -> outliers-2"
  
mos-cli stats-worker-assignments \
  "/tmp/crowdMOS.pkl" \
  "/tmp/worker-assignments.csv" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones" \
    "fast-workers & bad-workers -> no-headphones -> outliers-1" \
    "fast-workers & bad-workers -> no-headphones -> outliers-2"
  
mos-cli stats-algorithm-worker \
  "/tmp/crowdMOS.pkl" \
  "/tmp/algorithm-worker.csv" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones" \
    "fast-workers & bad-workers -> no-headphones -> outliers-1" \
    "fast-workers & bad-workers -> no-headphones -> outliers-2"
  
mos-cli stats-algorithm-sentences \
  "/tmp/crowdMOS.pkl" \
  "/tmp/algorithm-sentences.csv" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones" \
    "fast-workers & bad-workers -> no-headphones -> outliers-1" \
    "fast-workers & bad-workers -> no-headphones -> outliers-2"
  
mos-cli export-ground-truth \
  "/tmp/crowdMOS.pkl" \
  "/tmp/gt.csv" \
  -m \
    "fast-workers" \
    "bad-workers" \
    "fast-workers & bad-workers -> no-headphones" \
    "fast-workers & bad-workers -> no-headphones -> outliers-1" \
    "fast-workers & bad-workers -> no-headphones -> outliers-2"
  
mos-cli approve \
  "/tmp/crowdMOS.pkl" \
  "too_fast" \
  "/tmp/approve.csv" \
  --reason "good work"
  
mos-cli bonus \
  "/tmp/crowdMOS.pkl" \
  "too_fast" \
  10 \
  "really good work" \
  "/tmp/bonus.csv"
  
mos-cli reject \
  "/tmp/crowdMOS.pkl" \
  "too_fast" \
  "bad work" \
  "/tmp/reject.csv"
  