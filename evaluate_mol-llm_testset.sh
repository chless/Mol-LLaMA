DATA_CONFIG="/home/chanhui-lee/Mol-LLaMA/configs/mol_llm_testset/data_config.yaml"
TRAIN_CONFIG="/home/chanhui-lee/Mol-LLaMA/configs/mol_llm_testset/train_config.yaml"

python /home/chanhui-lee/Mol-LLaMA/evaluation_mol-llm_testset.py \
    --data_config_path "$DATA_CONFIG" \
    --train_config_path "$TRAIN_CONFIG"