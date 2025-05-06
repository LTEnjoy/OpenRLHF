deepspeed --num_gpus 1 \
   --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/Qwen2.5-0.5B-Instruct \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain /sujin/Models/Qwen/Qwen2.5-0.5B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
#   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --load_checkpoint \
#   --use_wandb {wandb_token}