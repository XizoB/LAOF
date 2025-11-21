


# Build Datasets
## LangSAM
```
git clone https://github.com/luca-medeiros/lang-segment-anything.git
```
## RAFT
```
git clone https://github.com/princeton-vl/RAFT.git
```
## 
```
python sample_datasets_with_opticalflow_sam_masknums.py env_name="bigfish" exp_name="default/bigfish_default"
python convert_data_rlds_masknums.py --env_name bigfish_0.01
python convert_data_rlds_masknums.py --env_name bigfish_0.01_noaction --train_shards 30
```
# Stage1
## ############# 1 LAPO idm ###############
```
python stage1_idm.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```

## ############# 2 CoMo idmres ###############
```
python stage1_idmres.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres"
```

## ############# 3 CoMo w/ OF idmres ###############
```
python stage1_idmres_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres_flowdecoder"
```

## ############# 4 LAOF ###############
```
python stage1_idm_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowdecoder"
```

## ############# 5 LAOM-Action ###############
```
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.01_noaction"
```
## ############# 6 LAOF-Action ###############
```
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.01_noaction"
```


# Stage2
```
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```


# Stage3
```bash
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
```
