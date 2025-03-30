# WhatsApp-Agent-Neural_maze-

```
conda create -n whatsapp python=3.12 -y
conda activate whatsapp
pip install -r requriements.txt
python setup.py install
```


## Use of Model
- Groq -> Text Generation , Image Understanding  , STT Module(whisper model)
- qdrant -> for long term memory ( used to store chats in form of vectors)
- Elevenlabs -> used to TTS( text to speech ) module
- together.ai -> for image generation ( we use flux model )
