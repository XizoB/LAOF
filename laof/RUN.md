# Build Datasets

## Prerequisites

First, install the following two libraries:

### LangSAM
https://github.com/luca-medeiros/lang-segment-anything.git

### RAFT
https://github.com/princeton-vl/RAFT.git

## Data Collection

Use the following command to collect the dataset:
```bash
python sample_datasets_with_opticalflow_sam_masknums.py env_name="bigfish" exp_name="default/bigfish_default"
```

Then convert it to RLDS format.

### With Action Labels
For datasets that include action labels:
```bash
python convert_data_rlds_masknums.py --env_name bigfish_0.01
```

### Without Action Labels
For datasets without action labels:
```bash
python convert_data_rlds_masknums.py --env_name bigfish_0.01_noaction --train_shards 30
```

---

# Stage 1: IDM + Action Decoder (No Action Supervision)

### 1. LAPO (Learning Action Priors from Observation)
```bash
python stage1_idm.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```

### 4. LAOF (Learning Action with Optical Flow Constraints)
```bash
python stage1_idm_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowdecoder"
```

### 5. LAOM-Action (Learning Action from Observation with Action Decoder)
```bash
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.01_noaction"
```

### 6. LAOF-Action (Learning Action with Optical Flow Constraints and Action Decoder)
```bash
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.01_noaction"
```

---

# Stage 2: Distillation

Train the behavior cloning model using distillation:
```bash
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```

---

# Stage 3: Fine-tuning

Fine-tune the model with decoding:
```bash
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```
