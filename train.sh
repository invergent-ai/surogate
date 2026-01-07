./csrc/build/train --model="Qwen/Qwen3-0.6B" \
    --train-file="output/train-*.bin" --eval-file="output/eval.bin" \
    --batch-size=2 --grad-accumulation=4 --gpus=1 --seq-len=2048 \
    --epochs=2 --eval-every-n-steps=0 --ckpt-interval=1000 \
    --out-dir="output" --lr=6e-4 --lr-schedule=cosine --warmup=150 --final-lr-fraction=0.1 \
    --recipe=bf16 --init-proj-to-zero --from-scratch