
python stage1_idm.py env_name="bigfish" exp_name="0_hidiv-mufag"
python stage2_bc.py env_name="bigfish" exp_name="0_hidiv-mufag"
python stage3_decoding.py env_name="bigfish" exp_name="0_hidiv-mufag"
python stage3_eval_agent_latest.py env_name="bigfish" exp_name="bigfish"
python eval_latent.py env_name="bigfish" exp_name="bigfish_0_kavig"




############ Genie
python eval_wm.py env_name="bigfish" exp_name="bigfish_0_tozuv"
python eval_idm_wm.py env_name="bigfish" exp_name="bigfish_0_tozuv"
python eval_idm_wm.py env_name="bigfish" exp_name="bigfish_0_kavig"
python stage2_irlv0.py env_name="bigfish" exp_name="bigfish_0_tozuv"
python stage2_irl.py env_name="bigfish" exp_name="bigfish_0_tozuv"



############ Offline
python stage2_irl_dis.py env_name="bigfish" exp_name="bigfish_0_tozuv" stage2irl.lr=1e-4 stage_exp_name="actor_1e-5_bc_detach"
python stage2_irl_dis.py env_name="bigfish" exp_name="bigfish_0_tozuv" stage2irl.lr=1e-4 stage_exp_name="actor_1e-4_bc_detach"
python stage2_irl_dis.py env_name="bigfish" exp_name="bigfish_0_tozuv" stage2irl.lr=1e-4 stage_exp_name="actor_1e-5_bc"
python stage2_irl_dis.py env_name="bigfish" exp_name="bigfish_0_tozuv" stage2irl.lr=1e-4 stage_exp_name="actor_1e-4_bc"









############ Default ############
python stage1_idm.py env_name="bigfish" exp_name="bigfish_default"
python stage1_idm.py env_name="bossfight" exp_name="bossfight_default"
python stage1_idm.py env_name="chaser" exp_name="chaser_default"
python stage1_idm.py env_name="climber" exp_name="climber_default"
python stage1_idm.py env_name="coinrun" exp_name="coinrun_default"
python stage1_idm.py env_name="dodgeball" exp_name="dodgeball_default"
python stage1_idm.py env_name="heist" exp_name="heist_default"
python stage1_idm.py env_name="maze" exp_name="maze_default"
python stage1_idm.py env_name="miner" exp_name="miner_default"
python stage1_idm.py env_name="plunder" exp_name="plunder_default"
python stage1_idm.py env_name="leaper" exp_name="leaper_default"
python stage1_idm.py env_name="ninja" exp_name="ninja_default"


python stage2_bc.py env_name="bigfish" exp_name="default/bigfish_default"
python stage2_bc.py env_name="bigfish" exp_name="bigfish_default"
python stage2_bc.py env_name="bossfight" exp_name="bossfight_default"
python stage2_bc.py env_name="chaser" exp_name="chaser_default"
python stage2_bc.py env_name="climber" exp_name="climber_default"
python stage2_bc.py env_name="coinrun" exp_name="coinrun_default"
python stage2_bc.py env_name="dodgeball" exp_name="dodgeball_default"
python stage2_bc.py env_name="heist" exp_name="heist_default"
python stage2_bc.py env_name="maze" exp_name="maze_default"
python stage2_bc.py env_name="miner" exp_name="miner_default"
python stage2_bc.py env_name="plunder" exp_name="plunder_default"
python stage2_bc.py env_name="leaper" exp_name="leaper_default"
python stage2_bc.py env_name="ninja" exp_name="ninja_default"


python stage3_decoding.py env_name="bigfish" exp_name="bigfish_default"
python stage3_decoding.py env_name="bossfight" exp_name="bossfight_default"
python stage3_decoding.py env_name="chaser" exp_name="chaser_default"
python stage3_decoding.py env_name="climber" exp_name="climber_default"
python stage3_decoding.py env_name="coinrun" exp_name="coinrun_default"
python stage3_decoding.py env_name="dodgeball" exp_name="dodgeball_default"
python stage3_decoding.py env_name="heist" exp_name="heist_default"
python stage3_decoding.py env_name="maze" exp_name="maze_default"
python stage3_decoding.py env_name="miner" exp_name="miner_default"
python stage3_decoding.py env_name="plunder" exp_name="plunder_default"
python stage3_decoding.py env_name="leaper" exp_name="leaper_default"
python stage3_decoding.py env_name="ninja" exp_name="ninja_default"



############ One Image ############
python stage1_idm_one.py env_name="bigfish" exp_name="bigfish_oneimage"
python stage1_idm_one.py env_name="bossfight" exp_name="bossfight_oneimage"
python stage1_idm_one.py env_name="chaser" exp_name="chaser_oneimage"
python stage1_idm_one.py env_name="climber" exp_name="climber_oneimage"
python stage1_idm_one.py env_name="coinrun" exp_name="coinrun_oneimage"
python stage1_idm_one.py env_name="dodgeball" exp_name="dodgeball_oneimage"
python stage1_idm_one.py env_name="heist" exp_name="heist_oneimage"
python stage1_idm_one.py env_name="maze" exp_name="maze_oneimage"
python stage1_idm_one.py env_name="miner" exp_name="miner_oneimage"
python stage1_idm_one.py env_name="plunder" exp_name="plunder_oneimage"
python stage1_idm_one.py env_name="leaper" exp_name="leaper_oneimage"
python stage1_idm_one.py env_name="ninja" exp_name="ninja_oneimage"



python stage2_bc_one.py env_name="bigfish" exp_name="bigfish_oneimage"
python stage2_bc_one.py env_name="bossfight" exp_name="bossfight_oneimage"
python stage2_bc_one.py env_name="chaser" exp_name="chaser_oneimage"
python stage2_bc_one.py env_name="climber" exp_name="climber_oneimage"
python stage2_bc_one.py env_name="coinrun" exp_name="coinrun_oneimage"
python stage2_bc_one.py env_name="dodgeball" exp_name="dodgeball_oneimage"
python stage2_bc_one.py env_name="heist" exp_name="heist_oneimage"
python stage2_bc_one.py env_name="maze" exp_name="maze_oneimage"
python stage2_bc_one.py env_name="miner" exp_name="miner_oneimage"
python stage2_bc_one.py env_name="plunder" exp_name="plunder_oneimage"
python stage2_bc_one.py env_name="leaper" exp_name="leaper_oneimage"
python stage2_bc_one.py env_name="ninja" exp_name="ninja_oneimage"



python stage3_decoding_one.py env_name="bigfish" exp_name="bigfish_oneimage"
python stage3_decoding_one.py env_name="bossfight" exp_name="bossfight_oneimage"
python stage3_decoding_one.py env_name="chaser" exp_name="chaser_oneimage"
python stage3_decoding_one.py env_name="climber" exp_name="climber_oneimage"
python stage3_decoding_one.py env_name="coinrun" exp_name="coinrun_oneimage"
python stage3_decoding_one.py env_name="dodgeball" exp_name="dodgeball_oneimage"
python stage3_decoding_one.py env_name="heist" exp_name="heist_oneimage"
python stage3_decoding_one.py env_name="maze" exp_name="maze_oneimage"
python stage3_decoding_one.py env_name="miner" exp_name="miner_oneimage"
python stage3_decoding_one.py env_name="plunder" exp_name="plunder_oneimage"
python stage3_decoding_one.py env_name="leaper" exp_name="leaper_oneimage"
python stage3_decoding_one.py env_name="ninja" exp_name="ninja_oneimage"





############ Two Image ############
python stage1_idm_two.py env_name="bigfish" exp_name="bigfish_twoimage"
python stage1_idm_two.py env_name="bossfight" exp_name="bossfight_twoimage"
python stage1_idm_two.py env_name="chaser" exp_name="chaser_twoimage"
python stage1_idm_two.py env_name="climber" exp_name="climber_twoimage"
python stage1_idm_two.py env_name="coinrun" exp_name="coinrun_twoimage"
python stage1_idm_two.py env_name="dodgeball" exp_name="dodgeball_twoimage"
python stage1_idm_two.py env_name="heist" exp_name="heist_twoimage"
python stage1_idm_two.py env_name="maze" exp_name="maze_twoimage"
python stage1_idm_two.py env_name="miner" exp_name="miner_twoimage"
python stage1_idm_two.py env_name="plunder" exp_name="plunder_twoimage"
python stage1_idm_two.py env_name="leaper" exp_name="leaper_twoimage"
python stage1_idm_two.py env_name="ninja" exp_name="ninja_twoimage"



python stage2_bc_two.py env_name="bigfish" exp_name="bigfish_twoimage"
python stage2_bc_two.py env_name="bossfight" exp_name="bossfight_twoimage"
python stage2_bc_two.py env_name="chaser" exp_name="chaser_twoimage"
python stage2_bc_two.py env_name="climber" exp_name="climber_twoimage"
python stage2_bc_two.py env_name="coinrun" exp_name="coinrun_twoimage"
python stage2_bc_two.py env_name="dodgeball" exp_name="dodgeball_twoimage"
python stage2_bc_two.py env_name="heist" exp_name="heist_twoimage"
python stage2_bc_two.py env_name="maze" exp_name="maze_twoimage"
python stage2_bc_two.py env_name="miner" exp_name="miner_twoimage"
python stage2_bc_two.py env_name="plunder" exp_name="plunder_twoimage"
python stage2_bc_two.py env_name="leaper" exp_name="leaper_twoimage"
python stage2_bc_two.py env_name="ninja" exp_name="ninja_twoimage"



python stage3_decoding_two.py env_name="bigfish" exp_name="bigfish_twoimage"
python stage3_decoding_two.py env_name="bossfight" exp_name="bossfight_twoimage"
python stage3_decoding_two.py env_name="chaser" exp_name="chaser_twoimage"
python stage3_decoding_two.py env_name="climber" exp_name="climber_twoimage"
python stage3_decoding_two.py env_name="coinrun" exp_name="coinrun_twoimage"
python stage3_decoding_two.py env_name="dodgeball" exp_name="dodgeball_twoimage"
python stage3_decoding_two.py env_name="heist" exp_name="heist_twoimage"
python stage3_decoding_two.py env_name="maze" exp_name="maze_twoimage"
python stage3_decoding_two.py env_name="miner" exp_name="miner_twoimage"
python stage3_decoding_two.py env_name="plunder" exp_name="plunder_twoimage"
python stage3_decoding_two.py env_name="leaper" exp_name="leaper_twoimage"
python stage3_decoding_two.py env_name="ninja" exp_name="ninja_twoimage"



############ BC Anchoring ############
python stage2_bc_anchoring.py env_name="bigfish" exp_name="bigfish_anchoring"
python stage2_bc_anchoring.py env_name="bossfight" exp_name="bossfight_anchoring"
python stage2_bc_anchoring.py env_name="chaser" exp_name="chaser_anchoring"
python stage2_bc_anchoring.py env_name="climber" exp_name="climber_anchoring"
python stage2_bc_anchoring.py env_name="coinrun" exp_name="coinrun_anchoring"
python stage2_bc_anchoring.py env_name="dodgeball" exp_name="dodgeball_anchoring"
python stage2_bc_anchoring.py env_name="heist" exp_name="heist_anchoring"
python stage2_bc_anchoring.py env_name="maze" exp_name="maze_anchoring"
python stage2_bc_anchoring.py env_name="miner" exp_name="miner_anchoring"
python stage2_bc_anchoring.py env_name="plunder" exp_name="plunder_anchoring"
python stage2_bc_anchoring.py env_name="leaper" exp_name="leaper_anchoring"
python stage2_bc_anchoring.py env_name="ninja" exp_name="ninja_anchoring"


python stage3_decoding.py env_name="bigfish" exp_name="bigfish_anchoring"
python stage3_decoding.py env_name="bossfight" exp_name="bossfight_anchoring"
python stage3_decoding.py env_name="chaser" exp_name="chaser_anchoring"
python stage3_decoding.py env_name="climber" exp_name="climber_anchoring"
python stage3_decoding.py env_name="coinrun" exp_name="coinrun_anchoring"
python stage3_decoding.py env_name="dodgeball" exp_name="dodgeball_anchoring"
python stage3_decoding.py env_name="heist" exp_name="heist_anchoring"
python stage3_decoding.py env_name="maze" exp_name="maze_anchoring"
python stage3_decoding.py env_name="miner" exp_name="miner_anchoring"
python stage3_decoding.py env_name="plunder" exp_name="plunder_anchoring"
python stage3_decoding.py env_name="leaper" exp_name="leaper_anchoring"
python stage3_decoding.py env_name="ninja" exp_name="ninja_anchoring"




############ BC Anchoring_0.1 ############
python stage2_bc_anchoring.py env_name="bigfish" exp_name="bigfish_anchoring_0.1"
python stage2_bc_anchoring.py env_name="bossfight" exp_name="bossfight_anchoring_0.1"
python stage2_bc_anchoring.py env_name="chaser" exp_name="chaser_anchoring_0.1"
python stage2_bc_anchoring.py env_name="climber" exp_name="climber_anchoring_0.1"
python stage2_bc_anchoring.py env_name="coinrun" exp_name="coinrun_anchoring_0.1"
python stage2_bc_anchoring.py env_name="dodgeball" exp_name="dodgeball_anchoring_0.1"
python stage2_bc_anchoring.py env_name="heist" exp_name="heist_anchoring_0.1"
python stage2_bc_anchoring.py env_name="maze" exp_name="maze_anchoring_0.1"
python stage2_bc_anchoring.py env_name="miner" exp_name="miner_anchoring_0.1"
python stage2_bc_anchoring.py env_name="plunder" exp_name="plunder_anchoring_0.1"
python stage2_bc_anchoring.py env_name="leaper" exp_name="leaper_anchoring_0.1"
python stage2_bc_anchoring.py env_name="ninja" exp_name="ninja_anchoring_0.1"


python stage3_decoding.py env_name="bigfish" exp_name="bigfish_anchoring_0.1"
python stage3_decoding.py env_name="bossfight" exp_name="bossfight_anchoring_0.1"
python stage3_decoding.py env_name="chaser" exp_name="chaser_anchoring_0.1"
python stage3_decoding.py env_name="climber" exp_name="climber_anchoring_0.1"
python stage3_decoding.py env_name="coinrun" exp_name="coinrun_anchoring_0.1"
python stage3_decoding.py env_name="dodgeball" exp_name="dodgeball_anchoring_0.1"
python stage3_decoding.py env_name="heist" exp_name="heist_anchoring_0.1"
python stage3_decoding.py env_name="maze" exp_name="maze_anchoring_0.1"
python stage3_decoding.py env_name="miner" exp_name="miner_anchoring_0.1"
python stage3_decoding.py env_name="plunder" exp_name="plunder_anchoring_0.1"
python stage3_decoding.py env_name="leaper" exp_name="leaper_anchoring_0.1"
python stage3_decoding.py env_name="ninja" exp_name="ninja_anchoring_0.1"





############ Traindata_Random Horiozn1 Default ############
python stage1_idm_onehorizon.py env_name="bigfish" data_type="random" exp_name="train_random/default/bigfish_default_datarandom"
python stage1_idm_onehorizon.py env_name="bossfight" data_type="random" exp_name="train_random/default/bossfight_default_datarandom"
python stage1_idm_onehorizon.py env_name="chaser" data_type="random" exp_name="train_random/default/chaser_default_datarandom"
python stage1_idm_onehorizon.py env_name="climber" data_type="random" exp_name="train_random/default/climber_default_datarandom"
python stage1_idm_onehorizon.py env_name="coinrun" data_type="random" exp_name="train_random/default/coinrun_default_datarandom"
python stage1_idm_onehorizon.py env_name="dodgeball" data_type="random" exp_name="train_random/default/dodgeball_default_datarandom"
python stage1_idm_onehorizon.py env_name="heist" data_type="random" exp_name="train_random/default/heist_default_datarandom"
python stage1_idm_onehorizon.py env_name="maze" data_type="random" exp_name="train_random/default/maze_default_datarandom"
python stage1_idm_onehorizon.py env_name="miner" data_type="random" exp_name="train_random/default/miner_default_datarandom"
python stage1_idm_onehorizon.py env_name="plunder" data_type="random" exp_name="train_random/default/plunder_default_datarandom"
python stage1_idm_onehorizon.py env_name="leaper" data_type="random" exp_name="train_random/default/leaper_default_datarandom"
python stage1_idm_onehorizon.py env_name="ninja" data_type="random" exp_name="train_random/default/ninja_default_datarandom"


python stage2_bc_onehorizon.py env_name="bigfish" data_type="random" exp_name="train_random/default/bigfish_default_datarandom"
python stage2_bc_onehorizon.py env_name="bossfight" data_type="random" exp_name="train_random/default/bossfight_default_datarandom"
python stage2_bc_onehorizon.py env_name="chaser" data_type="random" exp_name="train_random/default/chaser_default_datarandom"
python stage2_bc_onehorizon.py env_name="climber" data_type="random" exp_name="train_random/default/climber_default_datarandom"
python stage2_bc_onehorizon.py env_name="coinrun" data_type="random" exp_name="train_random/default/coinrun_default_datarandom"
python stage2_bc_onehorizon.py env_name="dodgeball" data_type="random" exp_name="train_random/default/dodgeball_default_datarandom"
python stage2_bc_onehorizon.py env_name="heist" data_type="random" exp_name="train_random/default/heist_default_datarandom"
python stage2_bc_onehorizon.py env_name="maze" data_type="random" exp_name="train_random/default/maze_default_datarandom"
python stage2_bc_onehorizon.py env_name="miner" data_type="random" exp_name="train_random/default/miner_default_datarandom"
python stage2_bc_onehorizon.py env_name="plunder" data_type="random" exp_name="train_random/default/plunder_default_datarandom"
python stage2_bc_onehorizon.py env_name="leaper" data_type="random" exp_name="train_random/default/leaper_default_datarandom"
python stage2_bc_onehorizon.py env_name="ninja" data_type="random" exp_name="train_random/default/ninja_default_datarandom"


python stage3_decoding_onehorizon.py env_name="bigfish" data_type="random" exp_name="train_random/default/bigfish_default_datarandom"
python stage3_decoding_onehorizon.py env_name="bossfight" data_type="random" exp_name="train_random/default/bossfight_default_datarandom"
python stage3_decoding_onehorizon.py env_name="chaser" data_type="random" exp_name="train_random/default/chaser_default_datarandom"
python stage3_decoding_onehorizon.py env_name="climber" data_type="random" exp_name="train_random/default/climber_default_datarandom"
python stage3_decoding_onehorizon.py env_name="coinrun" data_type="random" exp_name="train_random/default/coinrun_default_datarandom"
python stage3_decoding_onehorizon.py env_name="dodgeball" data_type="random" exp_name="train_random/default/dodgeball_default_datarandom"
python stage3_decoding_onehorizon.py env_name="heist" data_type="random" exp_name="train_random/default/heist_default_datarandom"
python stage3_decoding_onehorizon.py env_name="maze" data_type="random" exp_name="train_random/default/maze_default_datarandom"
python stage3_decoding_onehorizon.py env_name="miner" data_type="random" exp_name="train_random/default/miner_default_datarandom"
python stage3_decoding_onehorizon.py env_name="plunder" data_type="random" exp_name="train_random/default/plunder_default_datarandom"
python stage3_decoding_onehorizon.py env_name="leaper" data_type="random" exp_name="train_random/default/leaper_default_datarandom"
python stage3_decoding_onehorizon.py env_name="ninja" data_type="random" exp_name="train_random/default/ninja_default_datarandom"




############ Traindata_Mixed Horiozn1 Default ############
python stage1_idm.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_datamixed"
python stage1_idm.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_datamixed"
python stage1_idm.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_datamixed"
python stage1_idm.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_datamixed"
python stage1_idm.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_datamixed"
python stage1_idm.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_datamixed"
python stage1_idm.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_datamixed"
python stage1_idm.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_datamixed"
python stage1_idm.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_datamixed"
python stage1_idm.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_datamixed"
python stage1_idm.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_datamixed"
python stage1_idm.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_datamixed"


python stage2_bc.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_datamixed"
python stage2_bc.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_datamixed"
python stage2_bc.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_datamixed"
python stage2_bc.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_datamixed"
python stage2_bc.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_datamixed"
python stage2_bc.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_datamixed"
python stage2_bc.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_datamixed"
python stage2_bc.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_datamixed"
python stage2_bc.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_datamixed"
python stage2_bc.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_datamixed"
python stage2_bc.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_datamixed"
python stage2_bc.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_datamixed"


python stage3_decoding.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_datamixed"
python stage3_decoding.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_datamixed"
python stage3_decoding.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_datamixed"
python stage3_decoding.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_datamixed"
python stage3_decoding.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_datamixed"
python stage3_decoding.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_datamixed"
python stage3_decoding.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_datamixed"
python stage3_decoding.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_datamixed"
python stage3_decoding.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_datamixed"
python stage3_decoding.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_datamixed"
python stage3_decoding.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_datamixed"
python stage3_decoding.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_datamixed"



############ Train_data_Mixed_Sample Default Horiozn1 ############
python stage1_idm.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_horizon1_datamixed"
python stage1_idm.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_horizon1_datamixed"
python stage1_idm.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_horizon1_datamixed"
python stage1_idm.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_horizon1_datamixed"
python stage1_idm.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_horizon1_datamixed"
python stage1_idm.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_horizon1_datamixed"
python stage1_idm.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_horizon1_datamixed"
python stage1_idm.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_horizon1_datamixed"
python stage1_idm.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_horizon1_datamixed"
python stage1_idm.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_horizon1_datamixed"
python stage1_idm.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_horizon1_datamixed"
python stage1_idm.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_horizon1_datamixed"


python stage2_bc.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_horizon1_datamixed"
python stage2_bc.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_horizon1_datamixed"
python stage2_bc.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_horizon1_datamixed"
python stage2_bc.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_horizon1_datamixed"
python stage2_bc.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_horizon1_datamixed"
python stage2_bc.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_horizon1_datamixed"
python stage2_bc.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_horizon1_datamixed"
python stage2_bc.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_horizon1_datamixed"
python stage2_bc.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_horizon1_datamixed"
python stage2_bc.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_horizon1_datamixed"
python stage2_bc.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_horizon1_datamixed"
python stage2_bc.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_horizon1_datamixed"


python stage3_decoding.py env_name="bigfish" data_type="mixed" exp_name="train_random/default/bigfish_default_horizon1_datamixed"
python stage3_decoding.py env_name="bossfight" data_type="mixed" exp_name="train_random/default/bossfight_default_horizon1_datamixed"
python stage3_decoding.py env_name="chaser" data_type="mixed" exp_name="train_random/default/chaser_default_horizon1_datamixed"
python stage3_decoding.py env_name="climber" data_type="mixed" exp_name="train_random/default/climber_default_horizon1_datamixed"
python stage3_decoding.py env_name="coinrun" data_type="mixed" exp_name="train_random/default/coinrun_default_horizon1_datamixed"
python stage3_decoding.py env_name="dodgeball" data_type="mixed" exp_name="train_random/default/dodgeball_default_horizon1_datamixed"
python stage3_decoding.py env_name="heist" data_type="mixed" exp_name="train_random/default/heist_default_horizon1_datamixed"
python stage3_decoding.py env_name="maze" data_type="mixed" exp_name="train_random/default/maze_default_horizon1_datamixed"
python stage3_decoding.py env_name="miner" data_type="mixed" exp_name="train_random/default/miner_default_horizon1_datamixed"
python stage3_decoding.py env_name="plunder" data_type="mixed" exp_name="train_random/default/plunder_default_horizon1_datamixed"
python stage3_decoding.py env_name="leaper" data_type="mixed" exp_name="train_random/default/leaper_default_horizon1_datamixed"
python stage3_decoding.py env_name="ninja" data_type="mixed" exp_name="train_random/default/ninja_default_horizon1_datamixed"




############ Train_data_Mixed_Sample Anchoring Horiozn1 ############
python stage2_bc_anchoring.py env_name="bigfish" data_type="mixed" exp_name="train_random/anchoring/bigfish_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="bossfight" data_type="mixed" exp_name="train_random/anchoring/bossfight_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="chaser" data_type="mixed" exp_name="train_random/anchoring/chaser_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="climber" data_type="mixed" exp_name="train_random/anchoring/climber_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="coinrun" data_type="mixed" exp_name="train_random/anchoring/coinrun_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="dodgeball" data_type="mixed" exp_name="train_random/anchoring/dodgeball_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="heist" data_type="mixed" exp_name="train_random/anchoring/heist_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="maze" data_type="mixed" exp_name="train_random/anchoring/maze_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="miner" data_type="mixed" exp_name="train_random/anchoring/miner_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="plunder" data_type="mixed" exp_name="train_random/anchoring/plunder_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="leaper" data_type="mixed" exp_name="train_random/anchoring/leaper_anchoring_horizon1_datamixed"
python stage2_bc_anchoring.py env_name="ninja" data_type="mixed" exp_name="train_random/anchoring/ninja_anchoring_horizon1_datamixed"


python stage3_decoding.py env_name="bigfish" data_type="mixed" exp_name="train_random/anchoring/bigfish_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="bossfight" data_type="mixed" exp_name="train_random/anchoring/bossfight_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="chaser" data_type="mixed" exp_name="train_random/anchoring/chaser_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="climber" data_type="mixed" exp_name="train_random/anchoring/climber_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="coinrun" data_type="mixed" exp_name="train_random/anchoring/coinrun_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="dodgeball" data_type="mixed" exp_name="train_random/anchoring/dodgeball_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="heist" data_type="mixed" exp_name="train_random/anchoring/heist_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="maze" data_type="mixed" exp_name="train_random/anchoring/maze_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="miner" data_type="mixed" exp_name="train_random/anchoring/miner_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="plunder" data_type="mixed" exp_name="train_random/anchoring/plunder_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="leaper" data_type="mixed" exp_name="train_random/anchoring/leaper_anchoring_horizon1_datamixed"
python stage3_decoding.py env_name="ninja" data_type="mixed" exp_name="train_random/anchoring/ninja_anchoring_horizon1_datamixed"






############ Train_data_Mixed_Sample Anchoring ############
python stage2_bc_anchoring.py env_name="bigfish" data_type="mixed" exp_name="train_random/anchoring/bigfish_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="bossfight" data_type="mixed" exp_name="train_random/anchoring/bossfight_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="chaser" data_type="mixed" exp_name="train_random/anchoring/chaser_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="climber" data_type="mixed" exp_name="train_random/anchoring/climber_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="coinrun" data_type="mixed" exp_name="train_random/anchoring/coinrun_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="dodgeball" data_type="mixed" exp_name="train_random/anchoring/dodgeball_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="heist" data_type="mixed" exp_name="train_random/anchoring/heist_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="maze" data_type="mixed" exp_name="train_random/anchoring/maze_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="miner" data_type="mixed" exp_name="train_random/anchoring/miner_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="plunder" data_type="mixed" exp_name="train_random/anchoring/plunder_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="leaper" data_type="mixed" exp_name="train_random/anchoring/leaper_anchoring_datamixed"
python stage2_bc_anchoring.py env_name="ninja" data_type="mixed" exp_name="train_random/anchoring/ninja_anchoring_datamixed"


python stage3_decoding.py env_name="bigfish" data_type="mixed" exp_name="train_random/anchoring/bigfish_anchoring_datamixed"
python stage3_decoding.py env_name="bossfight" data_type="mixed" exp_name="train_random/anchoring/bossfight_anchoring_datamixed"
python stage3_decoding.py env_name="chaser" data_type="mixed" exp_name="train_random/anchoring/chaser_anchoring_datamixed"
python stage3_decoding.py env_name="climber" data_type="mixed" exp_name="train_random/anchoring/climber_anchoring_datamixed"
python stage3_decoding.py env_name="coinrun" data_type="mixed" exp_name="train_random/anchoring/coinrun_anchoring_datamixed"
python stage3_decoding.py env_name="dodgeball" data_type="mixed" exp_name="train_random/anchoring/dodgeball_anchoring_datamixed"
python stage3_decoding.py env_name="heist" data_type="mixed" exp_name="train_random/anchoring/heist_anchoring_datamixed"
python stage3_decoding.py env_name="maze" data_type="mixed" exp_name="train_random/anchoring/maze_anchoring_datamixed"
python stage3_decoding.py env_name="miner" data_type="mixed" exp_name="train_random/anchoring/miner_anchoring_datamixed"
python stage3_decoding.py env_name="plunder" data_type="mixed" exp_name="train_random/anchoring/plunder_anchoring_datamixed"
python stage3_decoding.py env_name="leaper" data_type="mixed" exp_name="train_random/anchoring/leaper_anchoring_datamixed"
python stage3_decoding.py env_name="ninja" data_type="mixed" exp_name="train_random/anchoring/ninja_anchoring_datamixed"






############ Eval_latent ############
python eval_latent.py env_name="bigfish" exp_name="default/bigfish_default"
python eval_latent.py env_name="bossfight" exp_name="default/bossfight_default"
python eval_latent.py env_name="chaser" exp_name="default/chaser_default"
python eval_latent.py env_name="climber" exp_name="default/climber_default"
python eval_latent.py env_name="coinrun" exp_name="default/coinrun_default"
python eval_latent.py env_name="dodgeball" exp_name="default/dodgeball_default"
python eval_latent.py env_name="heist" exp_name="default/heist_default"
python eval_latent.py env_name="leaper" exp_name="default/leaper_default"
python eval_latent.py env_name="maze" exp_name="default/maze_default"
python eval_latent.py env_name="miner" exp_name="default/miner_default"
python eval_latent.py env_name="ninja" exp_name="default/ninja_default"
python eval_latent.py env_name="plunder" exp_name="default/plunder_default"


python eval_latent.py env_name="bigfish" exp_name="anchoring/bigfish_anchoring"
python eval_latent.py env_name="bossfight" exp_name="anchoring/bossfight_anchoring"
python eval_latent.py env_name="chaser" exp_name="anchoring/chaser_anchoring"
python eval_latent.py env_name="climber" exp_name="anchoring/climber_anchoring"
python eval_latent.py env_name="coinrun" exp_name="anchoring/coinrun_anchoring"
python eval_latent.py env_name="dodgeball" exp_name="anchoring/dodgeball_anchoring"
python eval_latent.py env_name="heist" exp_name="anchoring/heist_anchoring"
python eval_latent.py env_name="leaper" exp_name="anchoring/leaper_anchoring"
python eval_latent.py env_name="maze" exp_name="anchoring/maze_anchoring"
python eval_latent.py env_name="miner" exp_name="anchoring/miner_anchoring"
python eval_latent.py env_name="ninja" exp_name="anchoring/ninja_anchoring"
python eval_latent.py env_name="plunder" exp_name="anchoring/plunder_anchoring"



python eval_latent_one.py env_name="bigfish" exp_name="bigfish_oneimage"
python eval_latent_one.py env_name="bossfight" exp_name="bossfight_oneimage"
python eval_latent_one.py env_name="coinrun" exp_name="coinrun_oneimage"
python eval_latent_one.py env_name="chaser" exp_name="chaser_oneimage"
python eval_latent_one.py env_name="climber" exp_name="climber_oneimage"
python eval_latent_one.py env_name="dodgeball" exp_name="dodgeball_oneimage"
python eval_latent_one.py env_name="heist" exp_name="heist_oneimage"
python eval_latent_one.py env_name="leaper" exp_name="leaper_oneimage"
python eval_latent_one.py env_name="maze" exp_name="maze_oneimage"
python eval_latent_one.py env_name="miner" exp_name="miner_oneimage"
python eval_latent_one.py env_name="ninja" exp_name="ninja_oneimage"
python eval_latent_one.py env_name="plunder" exp_name="plunder_oneimage"




############ Eval_videos ############
python eval_agent_latest.py env_name="bigfish" exp_name="default/bigfish_default"
python eval_agent_latest.py env_name="bossfight" exp_name="default/bossfight_default"
python eval_agent_latest.py env_name="chaser" exp_name="default/chaser_default"
python eval_agent_latest.py env_name="climber" exp_name="default/climber_default"
python eval_agent_latest.py env_name="coinrun" exp_name="default/coinrun_default"
python eval_agent_latest.py env_name="dodgeball" exp_name="default/dodgeball_default"
python eval_agent_latest.py env_name="heist" exp_name="default/heist_default"
python eval_agent_latest.py env_name="leaper" exp_name="default/leaper_default"
python eval_agent_latest.py env_name="maze" exp_name="default/maze_default"
python eval_agent_latest.py env_name="miner" exp_name="default/miner_default"
python eval_agent_latest.py env_name="ninja" exp_name="default/ninja_default"
python eval_agent_latest.py env_name="plunder" exp_name="default/plunder_default"


python eval_agent_latest.py env_name="bigfish" exp_name="anchoring/bigfish_anchoring"
python eval_agent_latest.py env_name="bossfight" exp_name="anchoring/bossfight_anchoring"
python eval_agent_latest.py env_name="chaser" exp_name="anchoring/chaser_anchoring"
python eval_agent_latest.py env_name="climber" exp_name="anchoring/climber_anchoring"
python eval_agent_latest.py env_name="coinrun" exp_name="anchoring/coinrun_anchoring"
python eval_agent_latest.py env_name="dodgeball" exp_name="anchoring/dodgeball_anchoring"
python eval_agent_latest.py env_name="heist" exp_name="anchoring/heist_anchoring"
python eval_agent_latest.py env_name="leaper" exp_name="anchoring/leaper_anchoring"
python eval_agent_latest.py env_name="maze" exp_name="anchoring/maze_anchoring"
python eval_agent_latest.py env_name="miner" exp_name="anchoring/miner_anchoring"
python eval_agent_latest.py env_name="ninja" exp_name="anchoring/ninja_anchoring"
python eval_agent_latest.py env_name="plunder" exp_name="anchoring/plunder_anchoring"






############### idm_aloneflowdecoder_onemask ###############
python stage1_idm_aloneflowdecoder_onemask.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_aloneflowdecoder_onemask"


############### idm flow separate wox1 onemask ###############
python stage1_idm_wmflow_separate_wox1_onemask.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flow_separate_wox1_onemask"
python stage1_idm_wmflow_separate_wox1_onemask.py env_name="chaser" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/chaser_flow_separate_wox1_onemask"
python stage1_idm_wmflow_separate_wox1_onemask.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flow_separate_wox1_onemask"
python stage1_idm_wmflow_separate_wox1_onemask.py env_name="heist" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/heist_flow_separate_wox1_onemask"


############### idm flow vq ###############
python stage1_idm_wmflow.py env_name="bigfish" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/bigfish_flow"
python stage1_idm_wmflow.py env_name="dodgeball" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/dodgeball_flow"
python stage1_idm_wmflow.py env_name="maze" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/maze_flow"


############### idm flow shared ###############
python stage1_idm_wmflow_shared.py env_name="maze" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/maze_flow_shared"


############### idm flow separate ###############
python stage1_idm_wmflow_separate.py env_name="maze" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/maze_flow_separate"
python stage1_idm_wmflow_separate.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flow_separate"


################ idm flow separate wox1 idmres ###############
python stage1_idm_wmflow_separate_wox1_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flow_separate_wox1_idmres"


############### idm flow separate wox1 idmres decode ###############
python stage1_idm_wmflow_separate_wox1_decode.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_flow_separate_wox1_decode"


############### idm vq idmres decode ###############
python stage1_idm_decode_vq_idmres.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_decode_vq_idmres"


############### idm flow vq idmflow decode ###############
python stage1_idm_decode_vq_idmflow.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_decode_vq_idmflow"


############### idm flow vq idmflow sam decode ###############
python stage1_idm_decode_vq_idmflow_sam.py env_name="leaper" data_type="opticalflow_rlds" exp_name="opticalflow_rlds/leaper_decode_vq_idmflowsam"



