import speech_recognition as sr
import pygame

recognizer = sr.Recognizer()
mic = sr.Microphone()
CHUNK = 1024

pygame.mixer.init()

while True:
    with mic as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        text.strip().lower()

        if text == "done":
            break

        if text == "hello":
            print("Hello!!")

        if "play" in text:
            print("Playing...")
            pygame.mixer.music.load("sound.mp3")

        if "stop" in text:
            print("Stopping...")
            pygame.mixer.music.stop()

    except sr.UnknownValueError:
        print("Could not understand audio")

    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")