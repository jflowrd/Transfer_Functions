'''
Authors: B. Karlsen-Baeck, D. Quartullo and J. Flowerdew
Function to apply the cable transfer function to beam profile measurements.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.signal import find_peaks

def to_linear(x):
    r'''
    Converts a signal in dB to linear.

    :param x: numpy-array signal in dB
    :return: numpy-array signal in linear
    '''
    return 10**(x/20)

def to_complex(freq, magnitude, phase, plot = False):
    ''' This function is used to turn the magnitude and phase information for the 2023 TF (saved as a .csv)
    into an array of complex numbers. '''
    # find peaks in the sawtooth phase plot
    peaks, _ = find_peaks(phase, height=170)

    # Extract frequencies corresponding to the peaks
    peak_frequencies = freq[peaks]

    # Calculate the delta frequency between each pair of consecutive peaks
    delta_frequency = np.diff(peak_frequencies)

    print('delta freq.:', delta_frequency.mean())
    print('cable length:', 3e8/(delta_frequency.mean()))
    #print(len(freq))

    if plot:
        plt.plot(peak_frequencies[:-1], delta_frequency)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Delta Frequency [Hz]')
        plt.show()

    #Define parameters
    frequency_range = np.linspace(0, freq.max(), len(freq))
    jump_frequency = delta_frequency.mean()

    # Calculate sawtooth function
    sawtooth_phase = np.mod(-180 + 360 * (frequency_range / jump_frequency), 360) - 180

    # Subtract ideal sawtooth (add because sawtooth is already inverted)
    TF_phase = phase + sawtooth_phase

    # Scale between -180 and 180 and convert to radians
    TF_phase = [((TF_phase[i] + 180) % 360 - 180)* np.pi / 180 for i in range(len(TF_phase))]

    if plot:
        plt.plot(freq, TF_phase , label = 'Phase')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.legend()
        plt.show()    
    
    # return the TF as a complex number
    return magnitude*np.cos(TF_phase) + magnitude*np.sin(TF_phase)*1j


def apply_CTF(profiles, time_per_datapoint, bins_per_frame, fig_directory = "CTF_Figs\\", apply_inverse = True, apply_2014= False, apply_2021 = False, apply_2023 = False, apply_2024 = False, apply_pickup=False, dt_plot = 200, plot_spectra = False, plot_tf=False):

    """
    Applies the SPS cable transfer function on a measured waterfall.

    Will output figures of the filtered vs. the raw data and can select between two types of transfer functions:
    One was measured in 2014, and another one in 2021. Sadly, the one from 2021 has an artefact, wherefore the 2014 one is preferable
    The transfer function can be applied to transform the signal the beam feels (e.g. from simulation) to the measured data,
    or to transform from measurement to felt signal (this would be the inverse)

    Arguments:

        :arg profiles [FxF array]: Waterfall of profiles to apply the CTF to
        :arg time_per_datapoint [float]: Bin spacing for each profile, i.e. time between each measurement in an individal profile. In seconds.
        :arg bins_per_frame [int]: Number of bins in each profile/frame
        :arg fig_directory [str]: Relative directory to output the plots to
        :arg apply_inverse [bool]: Whether to apply inverse CTF (Measurement->Seen signal) or forward CTF (Seen signal -> Measurement)
        :arg dt_plot [int]: How often to plot. Will plot every dt_plot-th frame.
        :arg plot_spectra [bool]: Whether to plot the profile spectruma vs the CTF in frequency domain.

    """
    #this_directory = os.path.dirname(os.path.realpath(__file__))
    #fig_directory = os.path.join(this_directory, fig_directory)
    #os.makedirs(fig_directory, exist_ok=True)
    #os.makedirs(fig_directory + "transfer_function", exist_ok = True)

    if apply_inverse:
        applyTF = -1

    else:
        applyTF = 1

    bin_length = time_per_datapoint
    n_bins_per_frame = bins_per_frame
    measured_waterfall_a = np.copy(profiles)

    if applyTF!=0:
        # Definition of the raised_cosine_filter, H_RC values are between 0 and 1
        def raised_cosine_filter(cutoff_left, cutoff_right, freq_tf):
            H_RC = np.zeros(len(freq_tf))
            index_inbetween = np.where((freq_tf<=cutoff_right)&(freq_tf>=cutoff_left))[0]
            index_before = np.where(freq_tf<cutoff_left)[0]
            index_after = np.where(freq_tf>cutoff_right)[0]

            H_RC[index_before] = 1
            H_RC[index_after] = 0
            H_RC[index_inbetween] = (1+np.cos(np.pi/(cutoff_right-cutoff_left)*(freq_tf[index_inbetween]-cutoff_left)))/2

            return H_RC

        # Parameters
        if apply_2024:
            year_CTF = 2024

        elif apply_2023:
            year_CTF = 2023 # Note this data is just the Tomo line

        elif apply_2021:
            year_CTF = 2021 # 2014 (pre-LS2, preferable), 2021 (newest, Charis' one with fake dip)

        elif apply_2014:
            year_CTF = 2014

        else:
            year_CTF = 2014

        apply_raisedCos_filter = True
        cutoff_left = 2.40e9 # CTF reliable up to 2.5 GHz
        cutoff_right = 2.50e9
        plot_every_CTF = dt_plot
        plots_freq_domain = plot_spectra

        if applyTF == 1:
            print('APPLY DIRECT CTF, YEAR '+str(year_CTF)+', RC filter '+str(apply_raisedCos_filter))

        elif applyTF == -1:
            print('APPLY INVERSE CTF, YEAR '+str(year_CTF)+', RC filter '+str(apply_raisedCos_filter))

        # Load CTF
        try:
            data_tf = np.load('cables_transfer_function/'+str(year_CTF)+'/cableTF.npz')
            freq_tf = data_tf['freqArray']
            tf = data_tf['transfer']
        except:
            print('Extracting data from .dat file '+str(year_CTF))
            data_tf = pd.read_excel('tomo_xfer_part.xlsx', usecols=['freq (Hz)', 'S21 ampl (dB)', 'S21 (Ang)'])
            freq_tf = data_tf['freq (Hz)'].to_numpy()
            tf = to_complex(freq_tf, to_linear(data_tf['S21 ampl (dB)']).to_numpy(), data_tf['S21 (Ang)'].to_numpy(), plot=False)
        
        # Print CTF info
        print('CTF: f_max [GHz] = ', freq_tf[-1]/1e9)
        print('CTF: f[0] [Hz] = ', freq_tf[0])
        print('CTF: Min max Delta f [Hz] = '+str(min(np.diff(freq_tf)))+' '+str(max(np.diff(freq_tf))))
        print('CTF: Delta t [ns] = ', 1e9/(2*freq_tf[-1]))
        print('CTF: Min max t_max [ns] = '+str(1e9/max(np.diff(freq_tf)))+' '+str(1e9/min(np.diff(freq_tf))))

        # Construct the frequency array based on parameters of input profiles
        print('PROFILE: Delta t [ns] = ', bin_length*1e9)
        print('PROFILE: t_max [ns] = ', n_bins_per_frame*bin_length*1e9)

        n_fft = 10*n_bins_per_frame # n_bins_per_frame, next_regular(1000*n_bins_per_frame)
        d_fft = bin_length
        freq_spectrum = np.fft.rfftfreq(n_fft, d_fft)

        # Apply the pick up transfer function
        if apply_pickup:
            print('APPLYING PICKUP TRANSFER FUNCTION')
            if year_CTF == 2023:
                TF_pickup_data  = pd.read_csv('tf-apwl10-sig8v4.dat', sep="\s+", skiprows=4, names = ['Freq. [Hz]', 'Re', 'Im'])
                TF_pickup_freq = TF_pickup_data['Freq. [Hz]'].to_numpy()[:2990]
                TF_pickup_array = (TF_pickup_data['Re'].to_numpy() + TF_pickup_data['Im'].to_numpy() * 1j)[:2990]
                tf_pickup_fine = np.interp(freq_tf, TF_pickup_freq, TF_pickup_array)
                tf_new = tf*abs(tf_pickup_fine)
            else:
                # Extend the pick up from 6 GHz to 10 GHz
                TF_pickup_data  = pd.read_csv('tf-apwl10-sig8v4.dat', sep="\s+", skiprows=4, names = ['Freq. [Hz]', 'Re', 'Im'])
                TF_pickup_freq = TF_pickup_data['Freq. [Hz]'].to_numpy()
                TF_pickup_array = (TF_pickup_data['Re'].to_numpy() + TF_pickup_data['Im'].to_numpy() * 1j)
                freq_step = TF_pickup_freq[1] - TF_pickup_freq[0]
                additional_points = int(np.ceil((10e9 - TF_pickup_freq[-1]) / freq_step))
                new_freq = np.linspace(TF_pickup_freq[-1] + freq_step, 10e9, additional_points, endpoint=True)
                new_TF_pickup_array = np.full_like(new_freq, TF_pickup_array[-1])
                extended_TF_pickup_freq = np.concatenate((TF_pickup_freq, new_freq))
                extended_TF_pickup_array = np.concatenate((TF_pickup_array, new_TF_pickup_array))
                tf_pickup_fine = np.interp(freq_tf, extended_TF_pickup_freq, extended_TF_pickup_array)
                tf_new = tf*abs(tf_pickup_fine)
                print("Length of tf: ", len(tf))
                print("Length of extended_TF_pickup_array: ", len(extended_TF_pickup_array))
                print("Length of tf_pickup_fine: ", len(tf_pickup_fine))

            if plot_tf:
                plt.plot(freq_tf, abs(tf), label = 'cabel TF')
                plt.plot(freq_tf, abs(tf_pickup_fine), label = 'pickup TF')
                plt.plot(freq_tf, abs(tf_new)*10, label = 'cabel TF + pickup TF')
                plt.yscale('log',base=10)
                plt.legend() 
                plt.show()

            tf = tf_new
            #exit()

        # Interpolate the original CTF with the new frequency array       
        tf_interp = np.interp(freq_spectrum, freq_tf, tf)
        print("length of tf_interp: ", len(tf_interp))

        # Construct the raised cosine filter
        if apply_raisedCos_filter:
            H_RC = raised_cosine_filter(cutoff_left, cutoff_right, freq_spectrum)

        # Plot the original and interpolated CTFs
        plt.plot(freq_tf/1e9, np.abs(tf), color = 'b', label = 'TF')
        plt.plot(freq_spectrum/1e9, np.abs(tf_interp), color = 'r', label = 'TF interp')
        plt.plot(freq_spectrum/1e9, np.abs(tf_interp*H_RC), color = 'g', label = 'TF interp RC')
        plt.grid()
        plt.legend()
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Amplitude [a.u.]')
        plt.title(r'Year CTF: '+str(year_CTF)+', RC filter: '+str(apply_raisedCos_filter)+'\n'+
                  r'Cutoff left [GHz]: '+str(round(cutoff_left/1e9,2))+', cutoff right [GHz]: '+str(round(cutoff_right/1e9,2))+'\n'+
                  '# fft points = '+str(n_fft)
                  , fontsize=20, fontweight='bold')

        plt.savefig(fig_directory+'CTF_interp.png', bbox_inches='tight')
        plt.xlim(-0.05, 3.2)
        max_value_plot_CTF = np.max(np.abs(tf_interp)[(freq_spectrum/1e9>=0)&(freq_spectrum/1e9<=3.2)])
        plt.ylim(0, max_value_plot_CTF+0.05*max_value_plot_CTF)
        plt.axvline(cutoff_left/1e9, ls = '--', color = 'k')
        plt.axvline(cutoff_right/1e9, ls = '--', color = 'k')
        plt.savefig(fig_directory+'CTF_interp_zoom.png', bbox_inches='tight')

        plt.clf()

        # Apply CTF to profiles
        for i in range(measured_waterfall_a.shape[0]):
            # Compute the profile spectrum
            profileRAW = copy.deepcopy(measured_waterfall_a[i,:])
            profileRAW_spectrum = np.fft.rfft(profileRAW, n=n_fft)
            # Apply the direct or inverse transfer function

            if applyTF == 1:
                transformed_profile_freqDomain = profileRAW_spectrum*tf_interp

            elif applyTF == -1:
                transformed_profile_freqDomain = profileRAW_spectrum/tf_interp

            # Multiply by the RC filter
            if apply_raisedCos_filter:
                transformed_profile_freqDomain_RC = H_RC * transformed_profile_freqDomain

            else:
                transformed_profile_freqDomain_RC = transformed_profile_freqDomain

            # Derive the transformed profile
            transformed_profile = np.fft.irfft(transformed_profile_freqDomain_RC)[:measured_waterfall_a.shape[1]]
            measured_waterfall_a[i,:] = transformed_profile

            # Plots
            if i%plot_every_CTF==0:
                # Profiles
                ax = plt.gca()
                ax.plot(profileRAW, color = 'b', label = "CTF Raw")
                ax.plot(transformed_profile*np.max(profileRAW)/np.max(transformed_profile), color = 'r', label = "CTF Applied")
                ax.grid()
                plt.title(r'Turn '+str(i+1), fontsize=20, fontweight='bold')
                ax.set_xlabel('Bin number')
                ax.set_ylabel('Amplitude [a.u.]', color = 'b')
                plt.savefig(fig_directory+'transfer_function/profiles_'+str(i+1)+'.png', bbox_inches='tight')
                half_bucket_bins = int(5e-9/bin_length/2)
                ax.set_xlim(np.argmax(profileRAW)-half_bucket_bins, np.argmax(profileRAW)+half_bucket_bins)
                ax.legend()
                plt.savefig(fig_directory+'transfer_function/profilesZoom_'+str(i+1)+'.png', bbox_inches='tight')
                plt.clf()

                # Spectrum and CTF
                if plots_freq_domain:
                    ax = plt.gca()
                    ax2 = ax.twinx()
                    ax.plot(freq_spectrum/1e9, np.abs(profileRAW_spectrum), color = 'b', label = "Raw data")
                    ax2.plot(freq_spectrum/1e9, np.abs(tf_interp), color = 'r', label = "Applied CTF")
                    ax.grid()
                    ax2.grid()
                    plt.title(r'Turn '+str(i+1), fontsize=20, fontweight='bold')
                    ax.set_xlabel('Frequency [GHz]')
                    ax.set_ylabel('Amplitude [a.u.]', color = 'b')
                    ax2.set_ylabel('Amplitude [a.u.]', color = 'r')
                    ax.legend()
                    ax2.legend()

                    if apply_raisedCos_filter:
                        ax.axvline(cutoff_left/1e9, ls = '--', color='y')
                        ax.axvline(cutoff_right/1e9, ls = '--', color='y')

                    plt.savefig(fig_directory+'transfer_function/spectrumCTF_'+str(i+1)+'.png', bbox_inches='tight')
                    plt.clf()

                # Transformed profile in freq. domain without/with RC filter and CTF
                if plots_freq_domain:
                    ax = plt.gca()
                    ax2 = ax.twinx()
                    ax.plot(freq_spectrum/1e9, np.abs(transformed_profile_freqDomain), color = 'b')
                    ax.plot(freq_spectrum/1e9, np.abs(transformed_profile_freqDomain_RC), color = 'g')
                    ax2.plot(freq_spectrum/1e9, np.abs(tf_interp), color = 'r')
                    ax.grid()
                    ax2.grid()
                    plt.title(r'Turn '+str(i+1), fontsize=20, fontweight='bold')
                    ax.set_xlabel('Frequency [GHz]')
                    ax.set_ylabel('Amplitude [a.u.]')
                    ax2.set_ylabel('Amplitude [a.u.]', color = 'r')

                    if apply_raisedCos_filter:
                        ax.axvline(cutoff_left/1e9, ls = '--', color='y')
                        ax.axvline(cutoff_right/1e9, ls = '--', color='y')

                    plt.savefig(fig_directory+'transfer_function/transProfCFT_'+str(i+1)+'.png', bbox_inches='tight')
                    plt.clf()

                    # Zoom
                    if apply_raisedCos_filter:
                        margin_ind_freq_reduced = 1*(cutoff_right-cutoff_left)
                        ind_freq_reduced = np.where((freq_spectrum>=cutoff_left-margin_ind_freq_reduced)&
                                                    (freq_spectrum<=cutoff_right+margin_ind_freq_reduced))[0]

                        ax = plt.gca()
                        ax2 = ax.twinx()
                        ax.plot(freq_spectrum[ind_freq_reduced]/1e9,
                                np.abs(transformed_profile_freqDomain[ind_freq_reduced]), color = 'b')

                        ax.plot(freq_spectrum[ind_freq_reduced]/1e9,
                                np.abs(transformed_profile_freqDomain_RC[ind_freq_reduced]), color = 'g')

                        ax2.plot(freq_spectrum[ind_freq_reduced]/1e9,
                                 np.abs(tf_interp[ind_freq_reduced]), color = 'r')

                        plt.title(r'Turn '+str(i+1), fontsize=20, fontweight='bold')
                        ax.set_xlabel('Frequency [GHz]')
                        ax.set_ylabel('Amplitude [a.u.]')
                        ax2.set_ylabel('Amplitude [a.u.]', color = 'r')
                        ax.axvline(cutoff_left/1e9, ls = '--', color='y')
                        ax.axvline(cutoff_right/1e9, ls = '--', color='y')
                        ax.grid()
                        ax2.grid()
                        plt.savefig(fig_directory+'transfer_function/transProfCFT_zoom_'+str(i+1)+'.png', bbox_inches='tight')
                        plt.clf()

    return measured_waterfall_a