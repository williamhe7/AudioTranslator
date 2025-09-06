from googletrans import Translator
from googletrans import LANGUAGES
import edge_tts
import asyncio
import numpy as np
import miniaudio
import pyaudio
from faster_whisper import WhisperModel

def query_devices(pa):
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        print(f"{i}: {dev['name']} - Host API: {dev['hostApi']} - In: {dev['maxInputChannels']} Out: {dev['maxOutputChannels']}")

# sort voices
async def query_voices():
    voicesList = []
    voices = await edge_tts.voices.list_voices()
    for voice in voices:
        shortName = voice['ShortName']

        #continue if the voice is one of the restricted ones (microsoft doesnt allow to use)
        if "DragonHDLatestNeural" in shortName:
            continue

        abriv = ''
        for i in range(len(shortName)):
            if (shortName[i] != '-'):
                abriv = abriv + shortName[i]
            else:
                break

        if abriv == language:
            voicesList.append(shortName)
    return voicesList

async def translate(sentences):

    print(f"TRANSLATE: {sentences}")

    result = await translator.translate(sentences, dest=language)
    
    print(f"{result.origin} -> {result.text}")

    text = result.text
    return str(text)

async def text_to_speech(text, voice):

    communicate = edge_tts.Communicate(text, voice=voice)

    audio_chunks = []

    #get all the chunks in the stream into array
    #processing all together is better than processing by chunk
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    
    #join data into bytes
    mp3_data = b"".join(audio_chunks)
    
    #uncompress datafile to raw PCM samples (16 bit)
    decoded = miniaudio.decode(mp3_data)

    #turn into np array from byte array (keep 16 bit)
    samples = np.frombuffer(decoded.samples, dtype=np.int16)

    #convert stero to mono or else changes the sound and accuracy
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)
        samples = samples.mean(axis=1).astype(np.int16)

    #convert np integer array to floating point values
    #divde to normalize between -1.0, 1.0
    audio_np = samples.astype(np.float32) / 32768.0

    return audio_np, samples, decoded.sample_rate


def play_audio(stream, audio_data, samples, sample_rate):

    #write to stream for realtime
    stream.write(audio_data.tobytes())

async def capture_audio(stream, frames_per_buffer, audio_queue):
    while True:
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        audio = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        await audio_queue.put(audio)
        await asyncio.sleep(0.25)

async def process_audio(queue, voice, output_stream):
    while True:
        #await next chunk
        audio_chunk = await queue.get()
        segments, _ = model.transcribe(
            audio_chunk,
            task='transcribe',
            beam_size=1,
            vad_filter=True,
            temperature=0
        )
        text = " ".join([s.text for s in segments]).strip()

        #skip empty
        if not text:
            continue

        print(f"Transcribed: {text}")

        #translate    
        translated_text = await translate(text)
        print(f"Translated: {translated_text}")

        #tts
        audio_data, samples, decoded_sample_rate = await text_to_speech(translated_text, voice)
        play_audio(output_stream, audio_data=audio_data, samples=samples, sample_rate=decoded_sample_rate)

        await asyncio.sleep(0.25)

async def translate_loop(input_stream, output_stream, rate, frames_per_buffer, voice):

    await asyncio.gather(
        capture_audio(input_stream, frames_per_buffer, audio_queue),
        process_audio(audio_queue, voice, output_stream)
    )
    
# -----------------------------------------------------------

#init

#whisper
model = WhisperModel("small", device="cuda", compute_type="int8")

#translator object
translator = Translator()

#pyaudio object
pa = pyaudio.PyAudio()
rate = 16000
frames_per_buffer = 16000

#audio chunk queue
audio_queue = asyncio.Queue()

#device
query_devices(pa)

#we seem to have no microphone on this computer unfortunately, get the earbuds
#choose input (or default)
try:
    default_input_index = pa.get_default_input_device_info()['index']
    print(f"Default input device index: {default_input_index}: {pa.get_default_input_device_info()['name']}")
except OSError as e:
    print("Default input device index: No Default Input Device")
sound_input = int(input("Choose input (index): "))

#choose output (or default)
try:
    default_output_index = pa.get_default_output_device_info()['index']
    print(f"Default output device index: {default_output_index}: {pa.get_default_output_device_info()['name']}")
except OSError as e:
    print("Default output device index: No Default Output Device")
sound_output = int(input("Choose output (index): "))

# choose destination language
print(LANGUAGES)
#languages such as 'en', etc
language = input("Choose which language: ")
for key in LANGUAGES:
    if LANGUAGES[key] == language.lower():
        language = key

#choose voice
#get voices based on the dest lang
voicesList = asyncio.run(query_voices())
voice = voicesList[0]

#change voice
change_voice = input("Change Voice? (Y/N)\n")
if change_voice.lower() == "y":
    for i in range(len(voicesList)):
        print(voicesList[i])
    voice = input("Choose Voice:\n")

# start stream

input_stream = pa.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=frames_per_buffer,
                input_device_index=sound_input)

# FORMAT MUST BE PAFLOAT32 TO ALIGN WITH NP ARRAY
output_stream = pa.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=frames_per_buffer,
                output_device_index=sound_output)

asyncio.run(translate_loop(input_stream=input_stream, output_stream=output_stream, rate=rate, frames_per_buffer=frames_per_buffer, voice=voice))