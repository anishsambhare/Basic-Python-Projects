from gtts import gTTS
import os 

text = " Kasa ahes mitara ?"

language = 'mr'  

obj = gTTS(text=text, lang=language, slow=False)
obj.save("sample.mp3")

os.system("sample.mp3")