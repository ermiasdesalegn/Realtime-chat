# 3D Adventure Game with AI Chat

A 3D adventure game built with Python, OpenGL, and OpenAI's Realtime API featuring interactive NPCs with voice and text chat capabilities.

## Features

- **3D Environment**: Navigate a virtual office space with realistic furniture and NPCs
- **Dual Chat System**: 
  - Voice chat using OpenAI's Realtime API
  - Text chat with voice responses
- **Interactive NPCs**:
  - Sarah Chen (HR Director) - Helpful with employee matters
  - Michael Chen (CEO) - Visionary leader sharing company insights
- **Real-time Voice Processing**: Record, send, and receive voice messages
- **Smart Interruption**: Interrupt AI responses by speaking or typing

## Controls

- **Movement**: WASD keys to move around
- **Mouse**: Look around and rotate camera
- **Voice Chat**: `Ctrl+Space` to record voice messages
- **Text Chat**: Type and press `Enter` to send text messages
- **Exit Chat**: `Shift+Q` to end conversation with NPCs
- **Exit Game**: `Escape` to quit the game

## Requirements

- Python 3.8+
- OpenAI API key
- Microphone and speakers/headphones
- Windows, macOS, or Linux

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ermiasdesalegn/Realtime-chat.git
cd Realtime-chat
```

2. Install required packages:
```bash
pip install pygame PyOpenGL PyOpenGL_accelerate openai python-dotenv pyaudio websockets numpy
```

3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Make sure your microphone and speakers are working
2. Run the game:
```bash
python app.py
```
3. Press `Enter` at the menu screen to start
4. Walk up to NPCs to start conversations
5. Use `Ctrl+Space` for voice chat or type for text chat
6. Both input methods result in voice responses from NPCs

## Technical Details

- **3D Graphics**: OpenGL with pygame
- **Audio Processing**: PyAudio for real-time recording/playback
- **AI Integration**: OpenAI's GPT-4 and Realtime API
- **Voice Synthesis**: OpenAI's TTS with character-specific voices
- **Network**: WebSocket connection for real-time communication

## Project Structure

- `app.py` - Main game file with all classes and logic
- `.env` - Environment variables (create this file)
- `README.md` - This file

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI for providing the AI capabilities
- PyGame community for the gaming framework
- OpenGL for 3D rendering support
```

