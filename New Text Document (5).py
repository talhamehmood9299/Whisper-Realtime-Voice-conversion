import speech_recognition as sr
import streamlit as st
import pygame
import os
from pathlib import Path
from openai import OpenAI
from transformers import pipeline

client = OpenAI()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
pipe_2 = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
pygame.mixer.init()


def recognize_speech(min_duration, max_duration):
    recognizer = sr.Recognizer()
    text = ""

    with sr.Microphone() as source:
        st.write("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Capture audio input
        audio = recognizer.listen(source, timeout=max_duration)

    try:
        st.write("Processing...")
        # Use recognize_google method with show_all=True to get details about the recognized speech
        recognized = recognizer.recognize_google(audio, language="es-ES", show_all=True)

        if 'alternative' in recognized:
            for alt in recognized['alternative']:
                if 'transcript' in alt and min_duration <= len(alt['transcript']) <= max_duration:
                    text = alt['transcript']
                    break

        if text:
            st.write("You said:", text)
        else:
            st.write("Speech not within specified duration or not recognized.")
    except sr.UnknownValueError:
        st.write("Sorry, could not understand audio.")
    except sr.RequestError as e:
        st.write("Error:", e)

    return text


def spa_eng(text):
    text = pipe(text)[0]["translation_text"]
    print(text)
    return text


def eng_spa(text):
    text = pipe_2(text)[0]["translation_text"]
    return text


def get_completion(text):
    output_directory = Path("C:/Users/amanullah.awan/Desktop/New folder (3)")

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


# Main function
if __name__ == "__main__":

    st.title("Spanish Assistant")
    st.sidebar.header("Quick Responses")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
            st.chat_message("user").markdown(button_value)
            st.session_state.messages.append({"role": "user", "content": button_value})
            assistant_text = eng_spa(button_value)
            get_completion(assistant_text)
            file_path = "C:/Users/amanullah.awan/Desktop/New folder (3)/speech.mp3"
            play_audio(file_path)
            close_media_player()
            delete_file(file_path)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    prompt = st.chat_input("Write the Answer?")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        assistant_text = eng_spa(prompt)
        get_completion(assistant_text)
        file_path = "C:/Users/amanullah.awan/Desktop/New folder (3)/speech.mp3"
        play_audio(file_path)
        close_media_player()
        delete_file(file_path)
        first_query = recognize_speech(min_duration=3, max_duration=1000)
        first = spa_eng(first_query)
        response = f"{first}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})










