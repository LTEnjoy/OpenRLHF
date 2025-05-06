deepspeed --num_gpus 4 \
   --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --input_template $'User: {}\nAssistant: ' \
   --train_batch_size 8 \
   --micro_train_batch_size 2 \
   --max_samples 10000 \
   --pretrain /sujin/Models/Qwen/Qwen2.5-0.5B-Instruct \
   --save_path ./checkpoint/Qwen2.5-0.5B-Instruct \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
#   --load_checkpoint \
#   --use_wandb 6ed8c632f9676e109dad08100d9b636e69138eb7