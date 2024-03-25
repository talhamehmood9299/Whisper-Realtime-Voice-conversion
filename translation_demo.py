from translatepy.translators.google import GoogleTranslate
from TranscriptionWindow import TranscriptionWindow
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from nltk.tokenize import sent_tokenize
from tempfile import NamedTemporaryFile
from transformers import pipeline
import speech_recognition as sr
from sys import platform
from openai import OpenAI
from pathlib import Path
import streamlit as st
from queue import Queue
from time import sleep
import subprocess
import argparse
import pygame
import torch
import nltk
import io
import os

os.environ['OPENAI_API_KEY'] = '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'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = OpenAI()

pipe_2 = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


def eng_spa(text):
    text = pipe_2(text)[0]["translation_text"]
    return text


def get_completion(text):
    output_directory = Path("C:/Users/awais.nayar/Desktop/SpeechToText real time whisper/whisper_real_time_translation")

    # Specify the file path
    speech_file_path = output_directory / "speech.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    response.stream_to_file(speech_file_path)


def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def close_media_player():
    pygame.mixer.music.stop()
    pygame.mixer.quit()


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")


def finalize_audio_process(text):
    assistant_text = eng_spa(text)
    get_completion(assistant_text)
    file_path = "C:/Users/awais.nayar/Desktop/SpeechToText real time whisper/whisper_real_time_translation/speech.mp3"
    play_audio(file_path)
    close_media_player()
    delete_file(file_path)


def main():
    # ==================Streamlit App code Start=================#
    st.title("Spanish Assistant")
    st.sidebar.header("Quick Responses")
    main_container = st.container(height=600, border=True)
    placeholder = st.empty()

    button_options = {
        "Intro": "Hello, This is talha from AZZ Medical Associate. How may I assist you?",
        "DOB.": "May have your date of birth?",
        "First-name": "Can u please spell your first name",
        "Medication": "tell me the medication name",
        "Ask_Brief_description": "Can you give me a brief description of the problem",
        "Hold": "Hold on a second!"
    }

    # Create buttons for each button name
    for button_name in button_options.keys():
        if st.sidebar.button(button_name):
            button_value = button_options[button_name]
            finalize_audio_process(button_value)
            # st.write(button_value)

    prompt = st.chat_input("Write the Answer?")

    # placeholder.write(prompt)
    if prompt:
        finalize_audio_process(prompt)

    # ==================Streamlit App code End===================#
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="device to user for CTranslate2 inference",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--translation_lang", default='English',
                        help="Which language should we translate into.", type=str)
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--threads", default=0,
                        help="number of threads used for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)

    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    if args.model == "large":
        args.model = "large-v2"

    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model + ".en"

    translation_lang = args.translation_lang
    device = args.device
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = args.compute_type
    cpu_threads = args.threads

    nltk.download('punkt')
    audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    # window = TranscriptionWindow()

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                text = ""

                segments, info = audio_model.transcribe(temp_file)
                for segment in segments:
                    text += segment.text
                # text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                last_four_elements = transcription[-10:]
                result = ''.join(last_four_elements)
                sentences = sent_tokenize(result)
                # window.update_text(sentences, translation_lang)

                # ==================Streamlit App code Start=================#

                # showing messages in the container
                with main_container:
                    for line in transcription:
                        st.write(f"Patient: ", line)

                # ==================Streamlit App code End===================#
                # Clear the console to reprint the updated transcription.

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
