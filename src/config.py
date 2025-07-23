class Config:
    DATA_PATH = "../data/sample_data"
    OUTPUT_PATH = "../output"
    DATASET = "MoNuSAC" # MoNuSAC, PanNuke, CoNSeP_Tiled
    NUM_CLASSES = 5 # 5, 6, 8
    BATCH_SIZE = 32
    STAGE_EPOCH = 25
    PATIENCE = 25