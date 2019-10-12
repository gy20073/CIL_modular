fname="/home/yang/data/aws_data/mkz/2019-06-18_18-06-23/debug/intersect_futher/intersect_further.png"

bash img2video.sh $fname # outputs $fname".mp4"

python fake_pkl.py $fname".pkl"

python ../utils/eval_video_mkz.py \
  -exp_id "mm45_v4_SqnoiseShoulder_rfsv6_withTL_fixTL" \
  -short_id "fixTL" \
  -video_path $fname".mp4" \
  -pickle_path $fname".pkl" \
  -gpu 0 \
  -townid "01" \
  -middle_already_zoomed True

outname=$fname".mp4.fixTL.mp4"

ffmpeg -i $outname $fname".out%d.png"