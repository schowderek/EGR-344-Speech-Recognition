
import speech_recognition as sr
import numpy as np # type: ignore
import matplotlib.pyplot as plt
from scipy.io import wavfile  # For reading the audio file
import librosa 
import librosa.display
import pathlib
from scipy import signal

import keyboard
import pyaudio as pya
import wave as wv
import keyboard as kb
import time


# Define Spectrum Plotter


def plot_spectrogram(file_path):
    # Read the wav file
    sample_rate, data = wavfile.read(file_path)

# Check if the audio is stereo or mono
    if len(data.shape) == 2:
        # Convert stereo to mono by averaging channels
        data = np.mean(data, axis=1)

# Plot the spectrogram
    plt.show(block=False)
    plt.figure(figsize=(10, 6))
    plt.specgram(data, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap='viridis')
    
    plt.title("Spectrogram of the Audio File")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 7500)
    plt.colorbar(label="Intensity (dB)")
    plt.show()

def plot_mfcc(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plot MFCCs
    plt.show(block=False)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()

          
while 1:

    # Record the audio from the microphone
    aFormat = pya.paInt16 # Bits pers sample
    chan = 1 # Number of audio channels
    w_s = 44100 # Sample rate, Hertz
    chunk = 1024 # Store 1024 samples in the buffer
    frames = [] # Empty array to store audio amplitude
    name = 'audio.wav' #Filename

    audio = pya.PyAudio()

    # audio.open opens a file, sets the format, channel, sample rate. Input = True writes to a .wav file instead of read.
    # Frames per buffer sets the amount of audio samples that are processed and stored
    stream = audio.open(format = aFormat, channels = 1, rate = w_s, input = True, frames_per_buffer = chunk)

    print('Press the "e" key to begin recording. Press "e" again to stop recording')
    kb.wait('e') # Wait until user pressed the e key to start recording
    time.sleep(0.1) # Wait half a second before recording starts
    print(' ++ Recording ++')
    while(1): # Run code until interrupted

        try: # Run code until specific key press
            data = stream.read(chunk) # Record audio from stored chunks
            frames.append(data) # Audio data is saved to frames array
        except KeyboardInterrupt: # If a button is pressed
            break # Stop code instantly
        if keyboard.is_pressed('e'): # If the e button is pressed
            time.sleep(0.1) # Record an extra half second
            break # Stop code

    stream.stop_stream() # Stop recording audio
    stream.close() # Close recording audio
    audio.terminate() # Close audio system

    

    wFile = wv.open(name, 'wb')
    wFile.setnchannels(chan)
    wFile.setsampwidth(audio.get_sample_size(aFormat))
    wFile.setframerate(w_s)
    wFile.writeframes(b''.join(frames))
    wFile.close()



    dat, rate = librosa.load("audio.wav")
    time1 = np.arange(0,len(dat))/rate

    plt.figure(figsize=(10, 16))
    plt.subplot(2,1,1)
    plt.plot(time1,dat)
    plt.title('Raw Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2,1,2)
    freq = np.fft.fft(dat)
    plt.plot(abs(freq))
    plt.title('FFT Plot - Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0,5000)
    plt.ylabel('Amplitude')
    plt.show()




    with wv.open(name, 'r') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()


        # Read the audio data as a numpy array
        audio_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)

        # Reshape the audio data to have separate channels
        audio_data = audio_data.reshape((num_frames, num_channels))
 

    # Define filter parameters
    cutoff_frequency = 2000  # Hz
    filter_order = 4

    # Normalize the cutoff frequency
    nyquist_frequency = 0.5 * sample_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    # Design the filter
    b, a = signal.butter(filter_order, normalized_cutoff, btype='lowpass')  

    w, h = signal.freqz(b, a, fs=sample_rate)

    # Plot the magnitude responses
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.ylim(-100, 10)

    plt.grid(True)
    plt.show()

    filtered_audio = signal.filtfilt(b, a, audio_data, axis=0)


    with wv.open('filtered_audio.wav', 'w') as wav_file:
        wav_file.setparams((num_channels, 2, sample_rate, num_frames, 'NONE', 'not compressed'))
        wav_file.writeframes(filtered_audio.astype(np.int16).tobytes())

    #Usage

    # Create a recognizer instance
    r = sr.Recognizer()

    # Open the audio file
    with sr.AudioFile("filtered_audio.wav") as source:
        audio = r.record(source)

    # Recognize the speech

    try:
        text = r.recognize_google(audio)
        print("Recognized Text:", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    filter_dat,filter_rate = librosa.load("filtered_audio.wav")
    filter_time = np.arange(0,len(filter_dat))/filter_rate
    plt.figure(figsize=(10, 10))
    plt.subplot(2,1,1)
    plt.plot(filter_time,filter_dat)
    plt.title('Filtered Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2,1,2)
    plt.plot(time1,dat)
    plt.title('Raw Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')


    plot_spectrogram("audio.wav")
    plot_spectrogram("filtered_audio.wav")

    plot_mfcc("filtered_audio.wav")

