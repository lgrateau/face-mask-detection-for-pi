import pyttsx3
 
 
engine = pyttsx3.init()
engine.setProperty("voice", "com.apple.speech.synthesis.voice.amelie")
engine.say("metter votre masque")

engine.runAndWait()
engine.stop()
#voices = engine.getProperty('voices')
#for voice in voices:
#    print(voice, voice.id)
#    engine.setProperty('voice', voice.id)
#    engine.say("metter votre masque")
#    engine.runAndWait()
    
