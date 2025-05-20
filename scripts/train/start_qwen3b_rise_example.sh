set -e
set -u

WANDB_TOKEN=xxx
RUN_NAME=xxx
DATA_DIR=/path/to/your/data
MODEL_DIR=/path/to/your/model
SAVE_DIR=/path/to/your/output

mkdir -p .checkpoints/$RUN_NAME
mkdir -p $SAVE_DIR

# set http_proxy if needed

# ray start --head --num-cpus=8  --dashboard-port=8265  --dashboard-host=0.0.0.0

sleep 10

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
  "env_vars": {
      "HUGGING_FACE_HUB_TOKEN": "your_huggingface_token",
      "LM_HARNESS_CACHE_PATH": "cache",
      "VLLM_ATTENTION_BACKEND": "XFORMERS",
      "PYTHONUNBUFFERED": "1",
      "WANDB_API_KEY": "your_wandb_token",
  },
  "working_dir": "your_working_dir",
  "pip": ["latex2sympy2", "word2number", "timeout_decorator"]
  }' -- PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.prompt_key=prompt \
  data.train_batch_size=1024 \
  +data.critique_batch_size=128 \
  data.val_batch_size=1024 \
  data.max_prompt_length=6000 \
  data.max_response_length=3000 \
  actor_rollout_ref.model.path=$MODEL_DIR \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48000 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48000 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=9e-6 \
  critic.model.path=$MODEL_DIR \
  critic.model.use_remove_padding=True \
  critic.ppo_max_token_len_per_gpu=24000 \
  critic.forward_max_token_len_per_gpu=48000 \
  reward_model.reward_func_path=verl_utils/reward/reward_func.py \
  algorithm.kl_ctrl.kl_coef=0.01 \
  trainer.project_name=verl \
  trainer.experiment_name=$RUN_NAME \
  trainer.default_local_dir=$SAVE_DIR/$RUN_NAME \
  trainer.logger=['console','wandb'] \
  +trainer.val_before_train=False \
  +trainer.online_critique=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=96 \
  trainer.save_rollout=True \
  trainer.test_freq=8 \
  trainer.total_epochs=12 2>&1 | tee -a .checkpoints/$RUN_NAME/train.log

ray stop
