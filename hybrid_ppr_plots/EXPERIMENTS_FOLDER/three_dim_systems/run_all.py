"""
Run all experiments sequentially.
"""
import os

os.system("python exp_1_all_systems_big_res_flow.py")
os.system("python exp_2_all_systems_large_res_good_eps.py")
os.system("python exp_3_all_systems_sinus_large_res.py")
os.system("python exp_4_all_systems_small_res_good_eps.py")
os.system("python exp_5_lorenz_eps_model_effect_of_eps.py")
os.system("python exp_6_lorenz_eps_model_effect_of_r_dim.py")
os.system("python exp_7_lorenz_flow_effect_of_rdim.py")
os.system("python exp_8_lorenz_sinus_model_effect_of_rdim.py")
