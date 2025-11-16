SAMPLE_DIR="/home/wonjung/mobile/samples"
CKPT_PATH="/home/wonjung/mobile/output/gauss_awgn_L4_12b_v1_True.pt"
M_SEQ="0123"
SNR_LIST="0,5,10,15,20"

python sample_test.py \
  --ckpt "$CKPT_PATH" \
  --sample_dir "$SAMPLE_DIR" \
  --m "$M_SEQ" \
  --stages 4 \
  --bits 12 \
  --img_size 128 \
  --snrs "$SNR_LIST" \
  --out_dir "./sample_test_out"