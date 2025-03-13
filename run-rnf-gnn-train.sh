export DATASET_NAME=asia
export NUM_SAMPLES=10000
export RNF_DIM=50
export RNF_INTERMEDIATE_DIM=25
export RNF_INIT_METHOD=xavier_uniform
export NUM_HEADS=3
export NUM_EPOCHS=1
export LEARNING_RATE=0.001
export BATCH_SIZE=64
export EXPERIMENT_ID=asia_v1

python src/rnf_gnn.py \
-dataset_name=$DATASET_NAME \
-num_samples=$NUM_SAMPLES \
-rnf_dim=$RNF_DIM \
-rnf_intermediate_dim=$RNF_INTERMEDIATE_DIM \
-rnf_init_method=$RNF_INIT_METHOD \
-num_heads=$NUM_HEADS \
-num_epochs=$NUM_EPOCHS \
-learning_rate=$LEARNING_RATE \
-batch_size=$BATCH_SIZE \
-experiment_id=$EXPERIMENT_ID 
