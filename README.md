# song_search_python
This project implements a lightweight song-recognition system based on digital signal processing and audio fingerprinting. The system converts reference songs and short microphone recordings into compact frequency-time fingerprints, then identifies the most likely song by matching fingerprint pairs and applying time-offset voting. 

2. Recognition Approach
The final approach is a landmark-style audio fingerprinting pipeline. The same feature-extraction function is used for reference songs and for the query, ensuring that both sides are represented in the same coordinate system.
REFERENCE DATABASE
Audio file -> Mono/resample -> STFT -> dB spectrogram -> local peaks
          -> peak-pair fingerprints -> quantisation -> .npy + database.json

QUERY
Microphone -> query.wav -> same fingerprint pipeline -> query.npy

SEARCH
query.npy -> compare with fingerprints/*.npy -> time-offset votes -> ranked result
