
############### sample datasets with opticalflow and sam ###############
python sample_datasets_with_opticalflow_sam_512.py env_name="bigfish" exp_name="default/bigfish_default"
python sample_datasets_with_opticalflow_sam_512.py env_name="chaser" exp_name="default/chaser_default"
python sample_datasets_with_opticalflow_sam_512.py env_name="heist" exp_name="default/heist_default"
python sample_datasets_with_opticalflow_sam_512.py env_name="leaper" exp_name="default/leaper_default"


python sample_datasets_with_opticalflow_sam_512_black.py env_name="bigfish" exp_name="default/bigfish_default"
python sample_datasets_with_opticalflow_sam_512_black.py env_name="chaser" exp_name="default/chaser_default"
python sample_datasets_with_opticalflow_sam_512_black.py env_name="heist" exp_name="default/heist_default"
python sample_datasets_with_opticalflow_sam_512_black.py env_name="leaper" exp_name="default/leaper_default"

python sample_datasets_with_opticalflow_sam_masknums.py env_name="bigfish" exp_name="default/bigfish_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="chaser" exp_name="default/chaser_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="chaser" exp_name="default/chaser_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="dodgeball" exp_name="default/dodgeball_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="heist" exp_name="default/heist_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="leaper" exp_name="default/leaper_default"
python sample_datasets_with_opticalflow_sam_masknums.py env_name="maze" exp_name="default/maze_default"






# ============================================================
# BIGFISH SUMMARY 9
# ============================================================
# Train set: 877 trajectories, 574,566 total steps
# Test set:  89 trajectories, 58,073 total steps
# ------------------------------------------------------------
# TOTAL:     966 trajectories, 632,639 total steps
# ============================================================


# ============================================================
# CHASER SUMMARY
# ============================================================
# Train set: 3,600 trajectories, 820,762 total steps
# Test set:  300 trajectories, 68,605 total steps
# ------------------------------------------------------------
# TOTAL:     3,900 trajectories, 889,367 total steps
# ============================================================

# ============================================================
# HEIST SUMMARY
# ============================================================
# Train set: 10,800 trajectories, 356,685 total steps
# Test set:  1,200 trajectories, 39,063 total steps
# ------------------------------------------------------------
# TOTAL:     12,000 trajectories, 395,748 total steps
# ============================================================



# ============================================================
# LEAPER SUMMARY
# ============================================================
# Train set: 5,400 trajectories, 342,590 total steps
# Test set:  600 trajectories, 36,741 total steps
# ------------------------------------------------------------
# TOTAL:     6,000 trajectories, 379,331 total steps
# ============================================================






################ convert rlds data ###############
python convert_data_rlds_masknums.py --env_name bigfish --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper --train_shards 30
python convert_data_rlds_masknums.py --env_name heist --train_shards 30



python convert_data_rlds_masknums.py --env_name bigfish --train_shards 100
python convert_data_rlds_masknums.py --env_name chaser --train_shards 100
python convert_data_rlds_masknums.py --env_name leaper --train_shards 100
python convert_data_rlds_masknums.py --env_name heist --train_shards 100


python convert_data_rlds_masknums.py --env_name bigfish --train_shards 500
python convert_data_rlds_masknums.py --env_name heist --train_shards 500


# nums_ep_config = {
#     "bigfish": 100,
#     "chaser": 300,
#     "dodgeball": 400,
#     "heist": 600,
#     "leaper": 600,
#     "maze": 1000,
# }


python convert_data_rlds.py --env_name bigfish --train_shards 30
python convert_data_rlds.py --env_name dodgeball --train_shards 30
python convert_data_rlds.py --env_name maze --train_shards 30
python convert_data_rlds.py --env_name leaper --train_shards 30
python convert_data_rlds.py --env_name dodgeball --train_shards 30


########## leaper
python convert_data_rlds_masknums.py --env_name leaper_0.01
python convert_data_rlds_masknums.py --env_name leaper_0.01_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.02 
python convert_data_rlds_masknums.py --env_name leaper_0.02_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.03 
python convert_data_rlds_masknums.py --env_name leaper_0.03_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.04
python convert_data_rlds_masknums.py --env_name leaper_0.04_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.05
python convert_data_rlds_masknums.py --env_name leaper_0.05_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.06
python convert_data_rlds_masknums.py --env_name leaper_0.06_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.07
python convert_data_rlds_masknums.py --env_name leaper_0.07_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.08
python convert_data_rlds_masknums.py --env_name leaper_0.08_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.09
python convert_data_rlds_masknums.py --env_name leaper_0.09_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.1 --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.1_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.2 --train_shards 30
python convert_data_rlds_masknums.py --env_name leaper_0.2_noaction --train_shards 30


########## bigfish
python convert_data_rlds_masknums.py --env_name bigfish_0.01
python convert_data_rlds_masknums.py --env_name bigfish_0.01_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.02
python convert_data_rlds_masknums.py --env_name bigfish_0.02_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.03
python convert_data_rlds_masknums.py --env_name bigfish_0.03_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.04
python convert_data_rlds_masknums.py --env_name bigfish_0.04_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.05
python convert_data_rlds_masknums.py --env_name bigfish_0.05_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.06
python convert_data_rlds_masknums.py --env_name bigfish_0.06_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.07
python convert_data_rlds_masknums.py --env_name bigfish_0.07_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.08
python convert_data_rlds_masknums.py --env_name bigfish_0.08_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.09
python convert_data_rlds_masknums.py --env_name bigfish_0.09_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.1 --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.1_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.2 --train_shards 30
python convert_data_rlds_masknums.py --env_name bigfish_0.2_noaction --train_shards 30


########## chaser
python convert_data_rlds_masknums.py --env_name chaser_0.01
python convert_data_rlds_masknums.py --env_name chaser_0.01_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.02
python convert_data_rlds_masknums.py --env_name chaser_0.02_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.03
python convert_data_rlds_masknums.py --env_name chaser_0.03_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.04
python convert_data_rlds_masknums.py --env_name chaser_0.04_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.05
python convert_data_rlds_masknums.py --env_name chaser_0.05_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.06
python convert_data_rlds_masknums.py --env_name chaser_0.06_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.07
python convert_data_rlds_masknums.py --env_name chaser_0.07_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.08
python convert_data_rlds_masknums.py --env_name chaser_0.08_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.09
python convert_data_rlds_masknums.py --env_name chaser_0.09_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.1 --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.1_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.2 --train_shards 30
python convert_data_rlds_masknums.py --env_name chaser_0.2_noaction --train_shards 30



########## heist
python convert_data_rlds_masknums.py --env_name heist_0.01
python convert_data_rlds_masknums.py --env_name heist_0.01_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.02
python convert_data_rlds_masknums.py --env_name heist_0.02_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.03
python convert_data_rlds_masknums.py --env_name heist_0.03_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.04
python convert_data_rlds_masknums.py --env_name heist_0.04_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.05
python convert_data_rlds_masknums.py --env_name heist_0.05_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.06
python convert_data_rlds_masknums.py --env_name heist_0.06_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.07
python convert_data_rlds_masknums.py --env_name heist_0.07_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.08
python convert_data_rlds_masknums.py --env_name heist_0.08_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.09
python convert_data_rlds_masknums.py --env_name heist_0.09_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.1 --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.1_noaction --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.2 --train_shards 30
python convert_data_rlds_masknums.py --env_name heist_0.2_noaction --train_shards 30




############### 1 LAPO idm ###############
python stage1_idm.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
python stage1_idm.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser"
python stage1_idm.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper"
python stage1_idm.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist"



############### 2 CoMo idmres ###############
python stage1_idmres.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres"
python stage1_idmres.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_idmres"
python stage1_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_idmres"
python stage1_idmres.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_idmres"


############### 3 CoMo w/ OF idmres ###############
python stage1_idmres_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres_flowdecoder"
python stage1_idmres_flowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_idmres_flowdecoder"
python stage1_idmres_flowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_idmres_flowdecoder"
python stage1_idmres_flowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_idmres_flowdecoder"


############### 4 LAOF idm_flowdecoder ###############
python stage1_idm_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowdecoder"
python stage1_idm_flowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flowdecoder"
python stage1_idm_flowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flowdecoder"
python stage1_idm_flowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flowdecoder"


############### CLAM idm_actiondecode ###############
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/chaser_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/leaper_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/heist_actiondecoder_0.02_noaction"

################ CLAM idm_actiondecode_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.02_noaction"



############### CLAM idm_actiondecode ###############
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/bigfish_actiondecoder_1.0_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/chaser_actiondecoder_1.0_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/leaper_actiondecoder_1.0_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/heist_actiondecoder_1.0_noaction"

################ CLAM idm_actiondecode_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_1.0_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_1.0_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_1.0_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=1.0 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_1.0_noaction"


############### CLAM idm_actiondecode ###############
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.5_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/chaser_actiondecoder_0.5_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/leaper_actiondecoder_0.5_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/heist_actiondecoder_0.5_noaction"

################ CLAM idm_actiondecode_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.5_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.5_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.5_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.5 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.5_noaction"


############### CLAM idm_actiondecode ###############
python stage1_idm_actiondecode.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_actiondecode"
python stage1_idm_actiondecode.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_actiondecode"
python stage1_idm_actiondecode.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_actiondecode"
python stage1_idm_actiondecode.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_actiondecode"



############### 7 LAOF-Only(z) idm_aloneflowdecoder ###############
python stage1_idm_aloneflowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder"
python stage1_idm_aloneflowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder"
python stage1_idm_aloneflowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder"
python stage1_idm_aloneflowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder"


############### 8 LAOF-Only(z,o) idm_aloneflowdecoder ###############
python stage1_idm_aloneflowdecoder_zo.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder_zo"
python stage1_idm_aloneflowdecoder_zo.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder_zo"
python stage1_idm_aloneflowdecoder_zo.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder_zo"
python stage1_idm_aloneflowdecoder_zo.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder_zo"


############### 9 LAOF flowautoencoder ###############
python stage1_flowautoencoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowautoencoder"
python stage1_flowautoencoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flowautoencoder"
python stage1_flowautoencoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flowautoencoder"
python stage1_flowautoencoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flowautoencoder"


############### 10 LAOF-FDM idm_wmflow_shared ###############
python stage1_idm_wmflow_shared.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_wmflow_shared"
python stage1_idm_wmflow_shared.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_wmflow_shared"
python stage1_idm_wmflow_shared.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_wmflow_shared"
python stage1_idm_wmflow_shared.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_wmflow_shared"








# ############### LAOF-Only-IDMFLOW idmflow_aloneflowdecoder ###############
# python stage1_idmflow_aloneflowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmflow_aloneflowdecoder"
# python stage1_idmflow_aloneflowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_idmflow_aloneflowdecoder"
# python stage1_idmflow_aloneflowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_idmflow_aloneflowdecoder"
# python stage1_idmflow_aloneflowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_idmflow_aloneflowdecoder"








############### leaper Ratio CLAM idm_actiondecoder_noaction ###############
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/leaper_actiondecoder_0.01_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/leaper_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/leaper_actiondecoder_0.03_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/leaper_actiondecoder_0.04_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/leaper_actiondecoder_0.05_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/leaper_actiondecoder_0.06_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/leaper_actiondecoder_0.07_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/leaper_actiondecoder_0.08_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/leaper_actiondecoder_0.09_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/leaper_actiondecoder_0.1_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/leaper_actiondecoder_0.2_noaction"


############### leaper Ratio CLAM w/ OF idm_actiondecoder_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.01_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.03_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.04_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.05_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.06_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.07_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.08_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.09_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.1_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.2_noaction"




############### bigfish Ratio CLAM idm_actiondecoder_noaction ###############
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.01_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.03_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.04_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.05_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.06_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.07_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.08_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.09_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.1_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.2_noaction"


############### bigfish Ratio CLAM w/ OF idm_actiondecoder_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.01_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.03_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.04_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.05_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.06_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.07_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.08_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.09_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.1_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.2_noaction"



############### chaser Ratio CLAM idm_actiondecoder_noaction ###############
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/chaser_actiondecoder_0.01_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/chaser_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/chaser_actiondecoder_0.03_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/chaser_actiondecoder_0.04_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/chaser_actiondecoder_0.05_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/chaser_actiondecoder_0.06_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/chaser_actiondecoder_0.07_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/chaser_actiondecoder_0.08_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/chaser_actiondecoder_0.09_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/chaser_actiondecoder_0.1_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/chaser_actiondecoder_0.2_noaction"


############### chaser Ratio CLAM w/ OF idm_actiondecoder_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.01_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.03_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.04_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.05_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.06_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.07_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.08_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.09_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.1_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.2_noaction"





############### heist Ratio CLAM idm_actiondecoder_noaction ###############
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/heist_actiondecoder_0.01_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/heist_actiondecoder_0.02_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/heist_actiondecoder_0.03_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/heist_actiondecoder_0.04_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/heist_actiondecoder_0.05_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/heist_actiondecoder_0.06_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/heist_actiondecoder_0.07_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/heist_actiondecoder_0.08_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/heist_actiondecoder_0.09_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/heist_actiondecoder_0.1_noaction"
python stage1_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/heist_actiondecoder_0.2_noaction"


############### heist Ratio CLAM w/ OF idm_actiondecoder_flowdecoder_noaction ###############
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.01_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.02_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.03 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.03_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.04 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.04_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.05 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.05_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.06 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.06_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.07 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.07_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.08 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.08_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.09 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.09_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.1 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.1_noaction"
python stage1_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.2 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.2_noaction"

















######################################## CONTINUE TRAINING ########################################

############### 1 Continue LAPO idm ###############
python stage1_continue_idm.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/bigfish_continue"
python stage1_continue_idm.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/chaser_continue"
python stage1_continue_idm.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/leaper_continue"
python stage1_continue_idm.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/heist_continue"

############### 2 Continue CoMo idmres ###############
python stage1_continue_idmres.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/bigfish_continue_idmres"
python stage1_continue_idmres.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/chaser_continue_idmres"
python stage1_continue_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/leaper_continue_idmres"
python stage1_continue_idmres.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/heist_continue_idmres"

############### 3 Continue CoMo w/ OF idmres ###############
python stage1_continue_idmres_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/bigfish_continue_idmres_flowdecoder"
python stage1_continue_idmres_flowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/chaser_continue_idmres_flowdecoder"
python stage1_continue_idmres_flowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/leaper_continue_idmres_flowdecoder"
python stage1_continue_idmres_flowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/heist_continue_idmres_flowdecoder"

############### 4 Continue LAOF idm_flowdecoder ###############
python stage1_continue_idm_flowdecoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/bigfish_continue_flowdecoder"
python stage1_continue_idm_flowdecoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/chaser_continue_flowdecoder"
python stage1_continue_idm_flowdecoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/leaper_continue_flowdecoder"
python stage1_continue_idm_flowdecoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/heist_continue_flowdecoder"


############### 5 Continue CLAM  ###############
python stage1_continue_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/bigfish_continue_actiondecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/chaser_continue_actiondecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/leaper_continue_actiondecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/heist_continue_actiondecoder_0.02_noaction"


############### 6 Continue LAOF-Action  ###############
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/bigfish_continue_actiondecoder_flowdecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/chaser_continue_actiondecoder_flowdecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/leaper_continue_actiondecoder_flowdecoder_0.02_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.02 exp_name="opticalflow_rlds_continue/heist_continue_actiondecoder_flowdecoder_0.02_noaction"



############### 5 Continue CLAM  ###############
python stage1_continue_idm_actiondecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/bigfish_continue_actiondecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/chaser_continue_actiondecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/leaper_continue_actiondecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/heist_continue_actiondecoder_0.01_noaction"


############### 6 Continue LAOF-Action  ###############
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/bigfish_continue_actiondecoder_flowdecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/chaser_continue_actiondecoder_flowdecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/leaper_continue_actiondecoder_flowdecoder_0.01_noaction"
python stage1_continue_idm_actiondecoder_flowdecoder_noaction.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds_continue/heist_continue_actiondecoder_flowdecoder_0.01_noaction"




############### 7 Continue CALM  ###############
python stage1_continue_idm_actiondecode.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/bigfish_continue_actiondecoder"
python stage1_continue_idm_actiondecode.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/chaser_continue_actiondecoder"
python stage1_continue_idm_actiondecode.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/leaper_continue_actiondecoder"
python stage1_continue_idm_actiondecode.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds_continue/heist_continue_actiondecoder"



















######################################## START BC  ########################################
############### 1 LAPO idm ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist"



############### 2 CoMo idmres ###############
python stage2_bc_idmres.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres"
python stage2_bc_idmres.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_idmres"
python stage2_bc_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_idmres"
python stage2_bc_idmres.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_idmres"


############### 3 CoMo w/ OF idmres ###############
python stage2_bc_idmres.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_idmres_flowdecoder"
python stage2_bc_idmres.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_idmres_flowdecoder"
python stage2_bc_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_idmres_flowdecoder"
python stage2_bc_idmres.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_idmres_flowdecoder"


############### 4 LAOF idm_flowdecoder ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowdecoder"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flowdecoder"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flowdecoder"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flowdecoder"


############### CLAM idm_actiondecode ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_0.01_noaction"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/chaser_actiondecoder_0.01_noaction"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/leaper_actiondecoder_0.01_noaction"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/heist_actiondecoder_0.01_noaction"

################ CLAM idm_actiondecode_flowdecoder_noaction ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/bigfish_actiondecoder_flowdecoder_0.01_noaction"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/chaser_actiondecoder_flowdecoder_0.01_noaction"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/leaper_actiondecoder_flowdecoder_0.01_noaction"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" ratio=0.01 exp_name="opticalflow_rlds/heist_actiondecoder_flowdecoder_0.01_noaction"


############### CLAM idm_actiondecode ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_actiondecode"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_actiondecode"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_actiondecode"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_actiondecode"








############### LAOF-FDM ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_wmflow_shared"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_wmflow_shared"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_wmflow_shared"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_wmflow_shared"

############### 7 LAOF-Only(z) idm_aloneflowdecoder ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder"

############### 8 LAOF-Only(z,o) idm_aloneflowdecoder ###############
python stage2_bc.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder_zo"
python stage2_bc.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder_zo"
python stage2_bc.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder_zo"
python stage2_bc.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder_zo"

############### 9 LAOF flowautoencoder ###############
python stage2_bc_autoencoder.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowautoencoder"
python stage2_bc_autoencoder.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flowautoencoder"
python stage2_bc_autoencoder.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flowautoencoder"
python stage2_bc_autoencoder.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flowautoencoder"



######################################## START RL  ########################################
############### LAOF-FDM ###############
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_wmflow_shared"
python stage3_decoding.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_wmflow_shared"
python stage3_decoding.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_wmflow_shared"
python stage3_decoding.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_wmflow_shared"

############### 7 LAOF-Only(z) idm_aloneflowdecoder ###############
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder"
python stage3_decoding.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder"
python stage3_decoding.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder"
python stage3_decoding.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder"

############### 8 LAOF-Only(z,o) idm_aloneflowdecoder ###############
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_aloneflowdecoder_zo"
python stage3_decoding.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder_zo"
python stage3_decoding.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_aloneflowdecoder_zo"
python stage3_decoding.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_aloneflowdecoder_zo"

############### 9 LAOF flowautoencoder ###############
python stage3_decoding.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flowautoencoder"
python stage3_decoding.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flowautoencoder"
python stage3_decoding.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flowautoencoder"
python stage3_decoding.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flowautoencoder"
