"""
The dictionary of system parameters including the hybrid parameters.
See readme.txt for changes compared to original results.
"""

dict_of_sys_params = {
    "Lorenz63": {"sigma": 10.0,
                 "rho": 28.0,
                 "beta": 8 / 3,
                 "dt": 0.05,  # New for better lyapunov
                 "lle": 0.9041,
                 "hybridparam": ["rho"],
                 "alias": "Lorenz-63"
                 },

    "Chen": {"a": 35.0,
             "b": 3.0,
             "c": 28.0,
             "dt": 0.02,  # New for better lyapunov
             "lle": 2.0138,
             "hybridparam": ["a"],
             "alias": "Chen"
             },

    "ChuaCircuit": {"alpha": 9.0,
                    "beta": 100 / 7,
                    "a": 8 / 7,
                    "b": 5 / 7,
                    "dt": 0.1,  # New for better lyapunov
                    "lle": 0.3380,
                    "hybridparam": ["alpha"],
                    "alias": "Chua-Circuit"
                    },

    "DoubleScroll": {"a": 0.8,
                     "dt": 0.3,  # Paper value (is good)
                     "lle": 0.04969,
                     "hybridparam": ["a"],
                     "alias": "Double-Scroll"},

    "Halvorsen": {"a": 1.27,
                  "dt": 0.05,  # Paper value (is good)
                  "lle": 0.7747,
                  "hybridparam": ["a"],
                  "alias": "Halvorsen"},

    "Roessler": {"a": 0.2,
                 "b": 0.2,
                 "c": 5.7,
                 "dt": 0.1,  # New for better lyapunov
                 "lle": 0.06915,
                 "hybridparam": ["c"],  # changed compared to original
                 "alias": "Rössler"
                 },

    "Rucklidge": {"kappa": 2.0,
                  "lam": 6.7,
                  "dt": 0.1,  # Paper value (is good)
                  "lle": 0.1912,
                  "hybridparam": ["kappa"],
                  "alias": "Rucklidge"
                  },

    "Thomas": {"b": 0.18,
               "dt": 0.3,  # New for better lyapunov
               "lle": 0.03801,
               "hybridparam": ["b"],
               "alias": "Thomas"
               },

    "WindmiAttractor": {"a": 0.7,
                        "b": 2.5,
                        "dt": 0.2,  # Paper value (is good)
                        "lle": 0.07986,
                        "hybridparam": ["a"],
                        "alias": "WINDMI"},
}
