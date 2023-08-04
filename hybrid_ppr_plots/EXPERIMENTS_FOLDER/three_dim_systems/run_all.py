"""
Run all experiments sequentially.
"""
import os

os.system("python all_systems_big_res_flow.py")
os.system("python all_systems_large_res_good_eps.py")
os.system("python all_systems_sinus_large_res.py")
os.system("python all_systems_small_res_good_eps.py")
os.system("python lorenz_eps_model_effect_of_eps.py")
os.system("python lorenz_eps_model_effect_of_r_dim.py")
os.system("python lorenz_flow_effect_of_rdim.py")
os.system("python lorenz_sinus_model_effect_of_rdim.py")
