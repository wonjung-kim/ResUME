# # Analog
# python test.py \
#   --model analog \
#   --stages 4 \
#   --bits 12 \
#   --norm \
#   --batch 1 \
#   --img_size 128 \
#   --m "0123" \
#   --ckpt "/home/wonjung/mobile/output/analog_rayleigh_L4_12b_v1_True_kld_0.01.pt" \
#   --out_dir "/home/wonjung/mobile/output/test_251201_analog_rayleigh" \
#   --target_cbr 0.00521

# # Digital
# python test.py \
#   --model gauss \
#   --stages 4 \
#   --bits 12 \
#   --norm \
#   --batch 1 \
#   --img_size 128 \
#   --m "0123" \
#   --ckpt "/mnt/data/wonjung/mobile/output/gauss_awgn_L4_12b_v1_loss_base_True_kld_0.01.pt" \
#   --out_dir "/mnt/data/wonjung/mobile/output/test_loss_abla/base" \
  
# # MobileNet
# python test.py \
#   --model mobilenet \
#   --stages 4 \
#   --bits 12 \
#   --norm \
#   --batch 1 \
#   --img_size 128 \
#   --m "0123" \
#   --ckpt "/mnt/data/wonjung/mobile/output/mobilenet_STE_awgn_L4_12b_v251230_True_kld_0.01.pt" \
#   --out_dir "/mnt/data/wonjung/mobile/output/test_cansvq_260101/mobilenet_STE" \

# MobileViT
python test.py \
  --model mobilevit \
  --stages 4 \
  --bits 12 \
  --norm \
  --batch 1 \
  --img_size 128 \
  --m "0123" \
  --ckpt "/mnt/data/wonjung/mobile/output/mobilevit_CANSVQ_awgn_L4_12b_v1_True_kld_0.01.pt" \
  --out_dir "/mnt/data/wonjung/mobile/output/test_cansvq_260101/mobilevit" \