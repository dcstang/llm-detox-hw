1. Setup (one-time)
pip install -U "torch>=2.1" "transformers>=4.45" "peft>=0.13" \
               "datasets>=2.20" "detoxify>=0.5" "torchao>=0.16" \
               "scikit-learn" "tqdm"

2. Data prep
python -m data_prep.build_pairs --out-dir data --max-rows 80000

3. SFT training
python -m src.detox_hw.train_sft --train data/sft.jsonl --out checkpoints/sft --epochs 1 --batch-size 4 --grad-accum 4

4. Task 1 eval
python -m tasks.task1_sft_eval --sft-dir checkpoints/sft --out submissions/task1_sft_eval.json

5. DPO training
python -m src.detox_hw.train_dpo --train data/dpo.jsonl --sft-dir checkpoints/sft --out checkpoints/dpo --epochs 1

6. Task 3 eval
python -m tasks.task3_dpo_eval --sft-dir checkpoints/sft --dpo-dir checkpoints/dpo --out submissions/task3_dpo_eval.json

7. RM training
python -m src.detox_hw.train_rm --train data/dpo.jsonl --out checkpoints/rm --val-fraction 0.1
python -m tasks.rm_eval --rm-dir checkpoints/rm --pairs data/dpo.jsonl

8. PPO runs (Tasks 6, 7, 8) — these use Docker (verl), the commands are long but already written in the README exactly as-is.

The only thing you can't run locally is the PPO steps — those require Docker + GPU + the verlai/verl image. Everything up to step 7 just needs Python + GPU.
