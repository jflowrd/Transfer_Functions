'''
Author: J. Flowerdew
Calculate transfer function from a reference input signal and measure output signal saved in .csv format
Note BQM measurement saturated in 2024 Q1 measurement and so TF cannot be used
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

data_type = 'BQM' #'MD_scope', 'BQM', 'ABWLM', 'Tomo', 'PS2SPS'
plot_individual = False

results_folder = 'results/' + data_type +'/'

# Define the range of data to read
a = 19900 #start just before signal
b = a + 1000 #choosing just a 1000 points to filter out noise
sample_factor = 1 # change sampling rate, i.e. sample_factor = 2 to halve the sampling rate
t_step = sample_factor*5.0e-11

# Function to read data from CSV file using pandas
def read_csv(file_path, data_type=data_type):
    if data_type == 'MD_scope':
        data = pd.read_csv(file_path, header=None, dtype={3: float, 4: float}, float_precision='high',encoding='cp1252')  # No header for column names
        x = data.iloc[a:b:sample_factor, 3].values # Assuming time is in column 1 (D) and amplitude is in column 2 (E)
        y = data.iloc[a:b:sample_factor, 4].values
    else:
        data = pd.read_csv(file_path, header=4, float_precision='high',encoding='cp1252' ) 
        x = data['Time'][a:b:sample_factor].values # Assuming time is in column 1 (D) and amplitude is in column 2 (E)
        y = data['Ampl'][a:b:sample_factor].values
    return x, y  

# Function to calculate transfer function
def calculate_transfer_function(input_data, output_data):
    input_fft = np.fft.rfft(input_data, 2*(len(input_data)-1))
    output_fft = np.fft.rfft(output_data, 2*(len(input_data)-1))
    transfer_function = output_fft / input_fft
    return transfer_function

# Specify the file paths for the input and output CSV files
if data_type == 'MD_scope':
    print("--- Analyzing MD scope ---")
    input_files = ["240216_152538_ref.csv", "240216_152631_ref.csv", "240216_152715_ref.csv", "240216_152756_ref.csv", "240216_152825_ref.csv", "240216_152853_ref.csv", "240216_152917_ref.csv", "240216_152948_ref.csv", "240216_153016_ref.csv"]
    output_files = ["240216_130403.csv", "240216_130453.csv",  "240216_130518.csv", "240216_130552.csv", "240216_130635.csv", "240216_130717.csv", "240216_130746.csv", "240216_130813.csv", "240216_130846.csv", "240216_130916.csv"]
else:
    if data_type == 'BQM' or data_type == 'bqm':
        print("--- Analyzing BQM ---")
        data_id = 1333
        
    elif data_type == 'ABWLM' or data_type == 'abwlm':
        print("--- Analyzing ABWLM ---")
        data_id = 1343

    elif data_type == 'PS2SPS' or data_type == 'ps2sps':
        print("--- Analyzing PS2SPS ---")
        data_id = 1353

    else:
        print("--- Analyzing Tomo ---")
        data_id = 1350

    input_files = [f"newscope_Ref--{i:05d}.csv" for i in range(10)]
    output_files = [f"C1--{data_id}--{i:05d}.csv" for i in range(10)]

# Read data from CSV files and calculate average transfer function
avg_input_pulse = np.zeros_like(read_csv(input_files[0])[1])
avg_output_pulse = np.zeros_like(read_csv(output_files[0])[1])
avg_transfer_function = np.zeros_like(avg_input_pulse, dtype=complex)

for input_file, output_file in zip(input_files, output_files):
    input_time, input_data = read_csv(input_file)
    output_time, output_data = read_csv(output_file)
    if plot_individual:
        plt.plot(input_time, input_data, label='Input')
        plt.plot(output_time, output_data, label='Output')
        plt.show()
    avg_input_pulse += input_data
    avg_output_pulse += output_data
    avg_transfer_function += calculate_transfer_function(input_data, output_data)

avg_input_pulse /= len(input_files)
avg_output_pulse /= len(output_files)
avg_transfer_function /= len(input_files)

# Plot the average input and output pulses
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(input_time, avg_input_pulse/max(avg_input_pulse), label='Average Input Pulse')
plt.plot(output_time, avg_output_pulse/max(avg_output_pulse), label='Average Output Pulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Average Input and Output Pulses')
plt.legend()
plt.grid(True)

# Plot the transfer function
frequency = np.fft.rfftfreq(2*(len(avg_transfer_function)-1), d = t_step)
avg_transfer_function = (avg_transfer_function)
plt.subplot(2, 1, 2)
plt.plot(frequency, abs(avg_transfer_function))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Transfer Function')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_folder + 'TF_' + data_type + '.png')
plt.show()


# Save the transfer function as a .npz file
np.savez(results_folder + 'cableTF_' + data_type + '.npz', frequency=frequency, transfer_function=avg_transfer_function)

###########################################################################################################

#Compare results to 2014 MD Scope TF
TF_data_2014 = np.load('cableTF_2014.npz')
TF_array_2014 = TF_data_2014['transfer']
freq_array_2014 = TF_data_2014['freqArray']
apply_pickup = False

print("Length of 2014: ", len(TF_array_2014))
print("Length of 2024: ", len(avg_transfer_function))

if apply_pickup:
    print('APPLYING PICKUP TRANSFER FUNCTION')
    pickup_data_tf  = pd.read_csv('tf-apwl10-sig8v4.dat', sep="\s+", skiprows=4, names = ['Freq. [Hz]', 'Re', 'Im'])
    pickup_freq_tf = pickup_data_tf['Freq. [Hz]'].to_numpy()
    tf_pickup = (pickup_data_tf['Re'].to_numpy() + pickup_data_tf['Im'].to_numpy() * 1j)
    freq_step = pickup_freq_tf[1] - pickup_freq_tf[0]    
    additional_points = int(np.ceil((10e9 - pickup_freq_tf[-1]) / freq_step))
    new_freq = np.linspace(pickup_freq_tf[-1] + freq_step, 10e9, additional_points, endpoint=True)
    new_TF_pickup_array = np.full_like(new_freq, tf_pickup[-1])
    extended_TF_pickup_freq = np.concatenate((pickup_freq_tf, new_freq))
    extended_TF_pickup_array = np.concatenate((tf_pickup, new_TF_pickup_array))
    tf_pickup_fine = np.interp(freq_array_2014, extended_TF_pickup_freq, extended_TF_pickup_array)
    TF_array_2014 = TF_array_2014*abs(tf_pickup_fine)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(frequency, abs(avg_transfer_function)/abs(avg_transfer_function[8]), label='2024')
plt.plot(freq_array_2014, abs(TF_array_2014)/abs(TF_array_2014[8]), label='2014')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Transfer Function')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frequency, abs(avg_transfer_function)/abs(avg_transfer_function[8]))
plt.plot(freq_array_2014, abs(TF_array_2014)/abs(TF_array_2014[8]))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.yscale('log',base=10)
#plt.xscale('log', base=10)
plt.title('Transfer Function')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_folder + 'TF_2014comparison_' + data_type + '.png')
plt.show()
