# launch the master node of ray in container
NUM_GPUS=4
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/sujin/PycharmProjects/OpenRLHF/run_env"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $NUM_GPUS \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node $NUM_GPUS \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $NUM_GPUS \
   --vllm_num_engines $NUM_GPUS \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --advantage_estimator reinforce \
   --pretrain /sujin/Models/Qwen/Qwen2.5-0.5B-Instruct \
   --reward_pretrain /sujin/Models/Qwen/Qwen2.5-0.5B-Instruct \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 4 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --max_samples 10000 \
   --generate_max_len 256 \
   --zero_stage 3 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-3 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep