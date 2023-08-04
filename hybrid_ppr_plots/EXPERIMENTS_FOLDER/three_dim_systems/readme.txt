Written: 04.08.2023 by Dennis Duncan

The following issues made the previous results (in the master's thesis)
not really usable, or less expressible:

1)
The Lyapunov exponent of the Rucklidge system, as given in Sprott, was far too low.
I figured, the best thing would be to calculate the lyapunov exponents my self.

2)
Corresponding to the first issue: The dt's in the original experiments were chosen
quite large leading to calculated LLEs which deviated quite a bit (15%) from the
literature values, and also to quite non-smooth attractors.
Consequently, I have adjusted the dt's for some systems in order to get good LLEs
corresponding to the Sprott literature values (assuming, the other lyaps are indeed
correct).
For the following systems, the time steps were adjusted leading to the following lles:
- Lorenz63: dt 0.1 -> 0.05, lle_calc = 0.9041 (sprott: 0.9056)
- Roessler: 0.2 -> 0.1, lle_calc = 0.06915 (sprott: 0.0714)
- Chen: 0.03 -> 0.02, lle_calc = 2.0138 (sprott: 2.0272)
- ChuaCircuit: 0.2 -> 0.1, lle_calc = 0.3380 (sprott: 0.3271)
- Thomas: 0.4 -> 0.3, lle_calc = 0.03801 (sprott: 0.0349)
- WINDMI: 0.2 -> 0.2 (same), lle_calc = 0.07986 (sprott: 0.0755)
- Rucklidge: 0.1 -> 0.1 (same), lle_calc = 0.1912 (sprott: 0.0643)
- Halvorsen: 0.05 -> 0.05 (same), lle_calc = 0.7747 (sprott: 0.7899)
- DoubleScroll: 0.3 -> 0.3 (good), lle_calc = 0.04969 (sprott: 0.0497)

The experiments should be repeated with these settings for dt and lle.

3)
One can get a much larger expressibility when also considering the fitted model
(corresponding to r_dim = 0). For the Roessler System it turns out, then when modifying
the "a" parameter, the fitted model is already as good as FH and OH (rdim = 500).
Thus, the high valid time in that case is not a result of the reservoir, but of the
accurately fitted model. When choosing the "c" parameter the fitted model is much worse.
Consequently, two changes were made here:
- The "c" parameter is now changed for the Roessler-Epsilon model.
- For each system, the fitted model accuracy is also calculated.


THIS FOLDER CONTAINS:
- The new experiment python files, adjusted to correct the above mentioned issues.
