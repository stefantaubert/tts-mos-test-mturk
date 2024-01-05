DS_PATH="/tmp/example.pkl"

# Initialize project using example.json
mos-cli init \
  "example.json" \
  "$DS_PATH"
# Output:
# Parsed 36 workers, 438 assignments and 2 x 4380 ratings for 4 algorithms and 120 files.

# Mask all approved assignments
mos-cli masks mask-assignments-by-status \
  "$DS_PATH" \
  "Approved" \
  "approved"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
# Assignments  438            0           140      140            0.0       31.963470  31.963470            438        298
#     Ratings 4380            0          1246     1246            0.0       32.022616  28.447489           3891       2645

# Mask all rejected assignments
mos-cli masks mask-assignments-by-status \
  "$DS_PATH" \
  "Rejected" \
  "rejected"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
# Assignments  438            0           168      168            0.0       38.356164  38.356164            438        270
#     Ratings 4380            0          1501     1501            0.0       38.576201  34.269406           3891       2390

# Mask all assignments which were done using headphones
mos-cli masks mask-assignments-by-device \
  "$DS_PATH" \
  "headphones" \
  "headphones"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
# Assignments  438            0           222      222            0.0       50.684932  50.684932            438        216
#     Ratings 4380            0          1964     1964            0.0       50.475456  44.840183           3891       1927
    
# Mask all workers that are male
mos-cli masks mask-workers-by-gender \
  "$DS_PATH" \
  "male" \
  "male"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
#     Workers   36            0            16       16            0.0       44.444444  44.444444             36         20
# Assignments  438            0           190      190            0.0       43.378995  43.378995            438        248
#     Ratings 4380            0          1689     1689            0.0       43.407864  38.561644           3891       2202

# Mask all workers that are in age group 18-29
mos-cli masks mask-workers-by-age-group \
  "$DS_PATH" \
  "18-29" \
  "18-29"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
#     Workers   36            0             9        9            0.0       25.000000  25.000000             36         27
# Assignments  438            0           111      111            0.0       25.342466  25.342466            438        327
#     Ratings 4380            0           982      982            0.0       25.237728  22.420091           3891       2909

# Merge masks not male and not 18-29
mos-cli masks merge-masks \
  "$DS_PATH" \
  '!male' '!18-29' \
  "group1"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
#     Workers   36           30             0       30      83.333333             0.0  83.333333              6          6
# Assignments  438          366             0      366      83.561644             0.0  83.561644             72         72
#     Ratings 4380         3253             0     3253      74.269406             0.0  74.269406            638        638

# Create mask for male workers in age group 18-29 by reversing the mask
mos-cli masks reverse-mask \
  "$DS_PATH" \
  "group1" "group1"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
#     Workers   36           30             6       36      83.333333           100.0 100.000000              6          0
# Assignments  438          366            72      438      83.561644           100.0 100.000000             72          0
#     Ratings 4380         3253           638     3891      74.269406           100.0  88.835616            638          0

# Mask workers that have less than 5 assignments
mos-cli masks mask-workers-by-assignments-count \
  "$DS_PATH" \
  5 \
  "workers-with-under-5-assignments" \
  --masks \
    "rejected"
# Output:
#        Type    #  #PrevMasked  #NewlyMasked  #Masked  #PrevMasked %  #NewlyMasked %  #Masked %  #PrevUnmasked  #Unmasked
#     Workers   36            0             4        4       0.000000       11.111111  11.111111             36         32
# Assignments  438          168            15      183      38.356164        5.555556  41.780822            270        255
#     Ratings 4380         1501           137     1638      34.269406        5.732218  37.397260           2390       2253
  
# Print MOS stats for all assignments that are not rejected and were done using headphones
mos-cli stats print-mos \
  "$DS_PATH" \
  --output "/tmp/example-stats.csv" \
  -m \
    'rejected' \
    '!headphones'

# Print MOS stats for all assignments that are not rejected, were done using headphones and are from group1
mos-cli stats print-mos \
  "$DS_PATH" \
  --output "/tmp/example-stats.csv" \
  -m \
    'rejected' \
    '!headphones' \
    '!group1'
