# python train.py \
#   --model="gauss" \
#   --stages=4 \
#   --bits=12 \
#   --batch=144 \
#   --version='1' \
#   --dataset='ImageNet' \
#   --norm \
#   --lr=1e-4 \
#   --m='0123' \
#   --kld_var=0.01 \
#   --target_cbr=0.015625 \
#   --epochs=200 \
#   --no-nsvq \

# python train.py \
#   --model="mobilenet" \
#   --stages=4 \
#   --bits=12 \
#   --batch=36 \
#   --version='1' \
#   --dataset='ImageNet' \
#   --norm \
#   --lr=1e-4 \
#   --m='0123' \
#   --kld_var=0.01 \
#   --target_cbr=0.015625 \
#   --epochs=200 \
#   --no-nsvq \

python train.py \
  --model="mobilevit" \
  --stages=4 \
  --bits=12 \
  --batch=36 \
  --version='1' \
  --dataset='ImageNet' \
  --norm \
  --lr=1e-3 \
  --m='0123' \
  --kld_var=0.01 \
  --epochs=200 \
  # --no-nsvq \
