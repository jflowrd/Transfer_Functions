# Transfer_Functions

''' Git repositary which collects the SPS transfer functions together '''

'README.txt' : Information about the data collected in 2024 [https://doi.org/10.5281/zenodo.10814568]

## Transfer function data

2014: MD line

2021: MD line (notch in Freq. spectrum)

2023: Tomo, ABWLM, PS2SPS

2024: MD, BQM*, Tomo, ABWLM, PS2SPS (*BQM signal saturated and so there is a notch in the freq. spectrum)

Transfer functions named cableTF.npz are for the MD scope line. Other transfer functions are named accordingly. (tf-apwl10-sig8v4.dat is the pickup TF).

More info: https://codimd.web.cern.ch/j8gUy4npRCmIkoaMmZjizQ?view

## Transfer function code

calculate_transfer_function.py: calculates the transfer function from .csv files collected in 2024

transfer_function_correction.py: function to apply transfer function (or inverse transfer function for simulation data) to waterfall data. 
