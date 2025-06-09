# Adventure Game
import os
# Set appropriate video driver for the platform
import platform
if platform.system() == 'Darwin':  # macOS
    os.environ['SDL_VIDEODRIVER'] = 'cocoa'
# Windows will use the default video driver
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
import sys
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import time
import asyncio
import websockets
import json
import pyaudio
import base64
import threading
import ssl

# Load environment variables
load_dotenv()
# Ensure OpenAI API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[OpenAI] API key not found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)
client = OpenAI(api_key=api_key)
print("[OpenAI] API key loaded successfully.")

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
screen = pygame.display.get_surface()

# Set up the camera and perspective
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Set up basic lighting
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 5, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

# Enable blending for transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Initial camera position
glTranslatef(0.0, 0.0, -5)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TILE_SIZE = 32
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# Game map
GAME_MAP = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W..................W",
    "W........N.........W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W....P.............W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
]

# Add these constants near the other constants
TITLE = "Venture Builder AI"
SUBTITLE = "Our Digital Employees"
MENU_BG_COLOR = (0, 0, 0)  # Black background
MENU_TEXT_COLOR = (0, 255, 0)  # Matrix-style green
MENU_HIGHLIGHT_COLOR = (0, 200, 0)  # Slightly darker green for effects

def draw_cube():
    vertices = [
        # Front face
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        # Back face
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ]
    
    surfaces = [
        [0, 1, 2, 3],  # Front
        [3, 2, 6, 5],  # Top
        [0, 3, 5, 4],  # Left
        [1, 7, 6, 2],  # Right
        [4, 5, 6, 7],  # Back
        [0, 4, 7, 1],  # Bottom
    ]
    
    glBegin(GL_QUADS)
    for surface in surfaces:
        glNormal3f(0, 0, 1)  # Simple normal for lighting
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

class AudioHandler:
    """Handles audio input and output using PyAudio for realtime voice chat."""
    def __init__(self):
        self.p = None
        self.stream = None
        self.audio_buffer = b''
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000  # Required by OpenAI Realtime API
        self.is_recording = False
        self.playback_stream = None
        self.is_playing = False
        self.playback_thread = None
        self._stop_event = threading.Event()  # Thread-safe stop signal
        self._initialized = False
        
        self._initialize_pyaudio()
        
    def _initialize_pyaudio(self):
        """Initialize PyAudio with error handling."""
        try:
            if self.p is not None:
                try:
                    self.p.terminate()
                except:
                    pass
            
            self.p = pyaudio.PyAudio()
            self._initialized = True
            
            # Print available audio devices
            print("\nAvailable Audio Devices:")
            for i in range(self.p.get_device_count()):
                dev_info = self.p.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']}")
                
        except Exception as e:
            print(f"[AudioHandler] Failed to initialize PyAudio: {e}")
            self._initialized = False
        


    def start_recording(self):
        """Start recording audio from microphone."""
        self.is_recording = True
        self.audio_buffer = b''
        
        print("[AudioHandler] Attempting to start recording...")
        
        # Ensure PyAudio is initialized
        if not self._initialized or self.p is None:
            print("[AudioHandler] PyAudio not initialized, attempting to reinitialize...")
            self._initialize_pyaudio()
            
        if not self._initialized:
            print("[AudioHandler] ERROR: Failed to initialize PyAudio!")
            self.is_recording = False
            return
        
        # List all available input devices
        input_devices = []
        try:
            for i in range(self.p.get_device_count()):
                try:
                    dev_info = self.p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        input_devices.append((i, dev_info))
                        print(f"[AudioHandler] Available input device {i}: {dev_info['name']} (channels: {dev_info['maxInputChannels']})")
                except:
                    continue
        except Exception as e:
            print(f"[AudioHandler] Error getting device list: {e}")
            self.is_recording = False
            return
        
        if not input_devices:
            print("[AudioHandler] ERROR: No input devices found!")
            self.is_recording = False
            return
        
        # Try devices in order: default first, then others
        devices_to_try = []
        
        # Try to get default device first
        try:
            default_input = self.p.get_default_input_device_info()
            devices_to_try.append((default_input['index'], default_input, "default"))
            print(f"[AudioHandler] Default input device: {default_input['name']}")
        except Exception as e:
            print(f"[AudioHandler] No default input device: {e}")
        
        # Add all other input devices
        for device_index, device_info in input_devices:
            if not any(d[0] == device_index for d in devices_to_try):
                devices_to_try.append((device_index, device_info, "fallback"))
        
        # Try each device until one works
        for device_index, device_info, device_type in devices_to_try:
            try:
                print(f"[AudioHandler] Trying {device_type} device {device_index}: {device_info['name']}")
                
                # Try different sample rates if the default doesn't work
                rates_to_try = [self.rate, 44100, 48000, 22050, 16000]
                
                for rate in rates_to_try:
                    try:
                        self.stream = self.p.open(
                            format=self.format,
                            channels=self.channels,
                            rate=rate,
                            input=True,
                            frames_per_buffer=self.chunk_size,
                            input_device_index=device_index
                        )
                        
                        # Test the stream by reading a small chunk
                        test_data = self.stream.read(128, exception_on_overflow=False)
                        
                        print(f"[AudioHandler] SUCCESS! Recording started with device {device_index} at {rate}Hz")
                        self.rate = rate  # Update rate to the working one
                        return
                        
                    except Exception as rate_error:
                        if self.stream:
                            try:
                                self.stream.close()
                            except:
                                pass
                            self.stream = None
                        continue
                        
            except Exception as device_error:
                print(f"[AudioHandler] Device {device_index} failed: {device_error}")
                continue
        
        # If we get here, no device worked
        print("[AudioHandler] ERROR: Failed to start recording with any available device!")
        print("[AudioHandler] Please check:")
        print("  1. Microphone is connected and enabled")
        print("  2. Windows Privacy Settings allow microphone access")
        print("  3. No other applications are using the microphone")
        self.is_recording = False
            
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("[AudioHandler] Stopped recording")
            
    def record_chunk(self):
        """Record a single chunk of audio."""
        if self.stream and self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_buffer += data
                return data
            except Exception as e:
                print(f"[AudioHandler] Error reading audio: {e}")
                return None
        return None
        
    def play_audio(self, audio_data):
        """Play audio data."""
        # Stop any existing playback first
        self.stop_playback()
        
        # Clear the stop event for new playback
        self._stop_event.clear()
        
        def play():
            local_stream = None
            try:
                self.is_playing = True
                local_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    output=True,
                    frames_per_buffer=1024
                )
                self.playback_stream = local_stream
                
                # Play audio in chunks to allow interruption
                chunk_size = 1024
                for i in range(0, len(audio_data), chunk_size):
                    if self._stop_event.is_set() or not self.is_playing:
                        print("[AudioHandler] Playback interrupted by stop event")
                        break
                    chunk = audio_data[i:i+chunk_size]
                    try:
                        local_stream.write(chunk)
                    except Exception as e:
                        print(f"[AudioHandler] Error writing chunk: {e}")
                        break
                
            except Exception as e:
                print(f"[AudioHandler] Error playing audio: {e}")
                print(f"[AudioHandler] ** IMPORTANT ** The AI responded but you can't hear it!")
                print(f"[AudioHandler] ** SOLUTION ** Check your speakers/headphones and Windows audio settings")
                print(f"[AudioHandler] ** Both Sarah and Michael ARE talking - you just can't hear them! **")
            finally:
                # Clean up stream safely
                if local_stream:
                    try:
                        local_stream.stop_stream()
                        local_stream.close()
                    except:
                        pass
                self.playback_stream = None
                self.is_playing = False
                print("[AudioHandler] Playback thread finished")
                
        # Wait for any existing thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self._stop_event.set()
            self.playback_thread.join(timeout=1.0)  # Wait max 1 second
        
        # Start new playback thread
        self.playback_thread = threading.Thread(target=play)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def stop_playback(self):
        """Stop any currently playing audio."""
        if self.is_playing:
            print("[AudioHandler] Stopping audio playback")
            self.is_playing = False
            self._stop_event.set()  # Signal the playback thread to stop
            
            # Wait for the thread to finish
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=2.0)  # Wait max 2 seconds
                if self.playback_thread.is_alive():
                    print("[AudioHandler] Warning: Playback thread did not stop in time")
            
            # Force close stream if still open
            if self.playback_stream:
                try:
                    self.playback_stream.stop_stream()
                    self.playback_stream.close()
                except:
                    pass
                self.playback_stream = None
        
    def cleanup(self):
        """Clean up resources."""
        print("[AudioHandler] Cleaning up...")
        if self.stream:
            self.stop_recording()
        self.stop_playback()
        
        # Make sure all threads are properly terminated
        if self.playback_thread and self.playback_thread.is_alive():
            self._stop_event.set()
            self.playback_thread.join(timeout=3.0)
        
        try:
            self.p.terminate()
        except Exception as e:
            print(f"[AudioHandler] Error during PyAudio cleanup: {e}")
        print("[AudioHandler] Cleanup complete")

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

class RealtimeDialogueSystem:
    def __init__(self):
        self.active = False
        self.ws = None
        self.audio_handler = AudioHandler()
        self.audio_buffer = b''  # Buffer for streaming audio responses
        self.is_speaking = False
        self.current_npc = None
        self.initial_player_pos = None
        
        # Voice input controls
        self.is_listening = False
        self.loop = None  # Will store the event loop
        
        # Text input controls
        self.user_input = ""
        self.input_active = False
        self.conversation_history = []  # For text-based conversations
        
        # UI setup
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            print("[RealtimeDialogueSystem] Font loaded successfully")
        except Exception as e:
            print("[RealtimeDialogueSystem] Font loading failed:", e)
            
        self.ui_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.ui_texture = glGenTextures(1)
        self.npc_message = ""
        
        # WebSocket setup
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-realtime-preview"  # Updated to supported realtime model
        
        # SSL configuration
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def start_conversation(self, npc_role="HR", player_pos=None):
        """Start a realtime voice conversation with the specified NPC."""
        print(f"[RealtimeDialogueSystem] Starting conversation with {npc_role}")
        print(f"[RealtimeDialogueSystem] Player position: {player_pos}")
        
        try:
            print("[RealtimeDialogueSystem] Setting up conversation...")
            self.active = True
            self.input_active = True  # Enable text input
            self.current_npc = npc_role
            self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
            
            # Initialize conversation history for text input
            self.conversation_history = [{
                "role": "system",
                "content": self.get_instructions_for_npc()
            }]
            
            print(f"[RealtimeDialogueSystem] Starting realtime dialogue with {npc_role}")
            print(f"[RealtimeDialogueSystem] API key present: {bool(api_key)}")
            print(f"[RealtimeDialogueSystem] Audio handler initialized: {bool(self.audio_handler)}")
            print(f"[RealtimeDialogueSystem] Current dialogue state: active={self.active}, is_listening={self.is_listening}, is_speaking={self.is_speaking}")
            
            # Start the async conversation in a separate thread
            print("[RealtimeDialogueSystem] Creating conversation thread...")
            self.conversation_thread = threading.Thread(target=self._run_async_conversation)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()
            print("[RealtimeDialogueSystem] Conversation thread started")
            
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Error starting conversation: {str(e)}")
            print(f"[RealtimeDialogueSystem] Error type: {type(e).__name__}")
            import traceback
            print(f"[RealtimeDialogueSystem] Traceback: {traceback.format_exc()}")
            self.active = False

    def _run_async_conversation(self):
        """Run the async conversation loop in a separate thread."""
        try:
            asyncio.run(self._conversation_loop())
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Conversation error: {e}")
            self.active = False

    async def _conversation_loop(self):
        """Main async conversation loop."""
        # Store the event loop for use in other threads
        self.loop = asyncio.get_running_loop()
        
        await self.connect_websocket()
        if not self.ws:
            print("[RealtimeDialogueSystem] Failed to connect, ending conversation")
            self.active = False
            return
            
        # Start listening for events
        listen_task = asyncio.create_task(self._listen_for_events())
        
        try:
            await listen_task
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Error in conversation loop: {e}")
        finally:
            if self.ws:
                await self.ws.close()
            self.audio_handler.cleanup()
            self.loop = None

    async def _listen_for_events(self):
        """Listen for events from the WebSocket."""
        try:
            async for message in self.ws:
                event = json.loads(message)
                await self._handle_event(event)
        except websockets.ConnectionClosed:
            print("[RealtimeDialogueSystem] WebSocket connection closed")
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Error listening for events: {e}")

    async def _handle_event(self, event):
        """Handle incoming events from the WebSocket server."""
        event_type = event.get("type")
        
        if event_type == "error":
            print(f"[RealtimeDialogueSystem] Error: {event['error']['message']}")
        elif event_type == "response.audio.delta":
            # Append audio data to buffer
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
            # Mark that AI is speaking
            if not self.is_speaking:
                self.is_speaking = True
                print("[RealtimeDialogueSystem] AI started speaking")
        elif event_type == "response.audio.done":
            # Play the complete audio response
            if self.audio_buffer:
                self.audio_handler.play_audio(self.audio_buffer)
                self.audio_buffer = b''
            self.is_speaking = False
            print("[RealtimeDialogueSystem] AI finished speaking")
        elif event_type == "response.text.delta":
            # Update displayed text
            if "delta" in event:
                self.npc_message += event["delta"]
        elif event_type == "response.text.done":
            # Text response complete
            print(f"[RealtimeDialogueSystem] NPC says: {self.npc_message}")
        elif event_type == "input_audio_buffer.speech_started":
            print("[RealtimeDialogueSystem] Speech detected")
        elif event_type == "input_audio_buffer.speech_stopped":
            print("[RealtimeDialogueSystem] Speech ended")
        elif event_type == "conversation.item.created":
            if event.get("item", {}).get("role") == "assistant":
                # Reset message for new response
                self.npc_message = ""

    async def send_audio_chunk(self, audio_data):
        """Send audio chunk to the realtime API."""
        if self.ws and audio_data:
            base64_chunk = base64.b64encode(audio_data).decode('utf-8')
            await self.send_event({
                "type": "input_audio_buffer.append",
                "audio": base64_chunk
            })

    async def commit_audio_buffer(self):
        """Commit the audio buffer and trigger AI response."""
        if self.ws:
            await self.send_event({"type": "input_audio_buffer.commit"})
            await self.send_event({"type": "response.create"})
            print("[RealtimeDialogueSystem] Audio sent, waiting for AI response...")

    async def cancel_ai_response(self):
        """Cancel the current AI response."""
        if self.ws:
            await self.send_event({"type": "response.cancel"})

    def toggle_recording(self):
        """Toggle recording on/off and send when stopping."""
        print(f"[RealtimeDialogueSystem] *** TOGGLE_RECORDING CALLED! *** Current state: {self.is_listening}")
        if self.is_listening:
            # Currently recording - stop and send
            print("[RealtimeDialogueSystem] Stopping recording...")
            self.stop_recording()
        else:
            # Not recording - start recording
            print("[RealtimeDialogueSystem] Starting recording...")
            self.start_recording()

    async def send_text_message(self, text_content):
        """Send a text message and get voice response via realtime API."""
        if self.ws:
            # Send the text as a conversation item
            await self.send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": text_content
                    }]
                }
            })
            
            # Trigger AI response
            await self.send_event({"type": "response.create"})
            print(f"[RealtimeDialogueSystem] Text message sent: {text_content}")

    def send_text_message_sync(self):
        """Synchronous wrapper to send text message."""
        if not self.user_input.strip():
            return
            
        print(f"[RealtimeDialogueSystem] User typed: {self.user_input}")
        
        # If AI is speaking, interrupt it first
        if self.is_speaking:
            print("[RealtimeDialogueSystem] Interrupting AI for text input...")
            self.interrupt_ai()
        
        # Store the message for history
        user_message = self.user_input.strip()
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Send via WebSocket if available, otherwise use fallback
        if self.loop and self.ws:
            asyncio.run_coroutine_threadsafe(
                self.send_text_message(user_message), 
                self.loop
            )
        else:
            # Fallback to direct OpenAI API if WebSocket not available
            self.send_text_fallback()
        
        # Clear input
        self.user_input = ""

    def send_text_fallback(self):
        """Fallback method to send text via OpenAI API and get audio response."""
        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=self.conversation_history,
                temperature=0.85,
                max_tokens=150
            )
            
            ai_message = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant", 
                "content": ai_message
            })
            
            # Convert AI response to speech
            self.text_to_speech(ai_message)
            
            # Update displayed message
            self.npc_message = ai_message
            
        except Exception as e:
            error_msg = "I apologize, but I'm having trouble connecting right now."
            self.npc_message = error_msg
            self.text_to_speech(error_msg)
            print(f"[RealtimeDialogueSystem] Text fallback error: {e}")

    def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS."""
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.get_tts_voice_for_npc(),
                input=text
            )
            
            # Play the audio response
            audio_data = response.content
            self.audio_handler.play_audio(audio_data)
            
        except Exception as e:
            print(f"[RealtimeDialogueSystem] TTS error: {e}")

    def get_tts_voice_for_npc(self):
        """Get TTS voice for current NPC."""
        if self.current_npc == "HR":
            return "nova"  # Female voice for HR Director Sarah
        else:  # CEO
            return "onyx"   # Male voice for CEO Michael

    def start_recording(self):
        """Start recording audio for voice input."""
        if not self.is_listening:
            print("[RealtimeDialogueSystem] Initializing recording...")
            # If AI is speaking, interrupt it first
            if self.is_speaking:
                print("[RealtimeDialogueSystem] Interrupting AI for voice recording...")
                self.interrupt_ai()
            
            # Also stop any lingering audio playback
            self.audio_handler.stop_playback()
            
            # Clear any current text input to focus on voice
            if self.user_input:
                print("[RealtimeDialogueSystem] Clearing text input to focus on voice...")
                self.user_input = ""
            
            self.is_listening = True
            print("[RealtimeDialogueSystem] Starting audio handler recording...")
            self.audio_handler.start_recording()
            print("[RealtimeDialogueSystem] Started recording...")
            
            # Start audio streaming in a separate thread
            print("[RealtimeDialogueSystem] Starting audio streaming thread...")
            self.audio_thread = threading.Thread(target=self._stream_audio_continuously)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("[RealtimeDialogueSystem] Audio streaming thread started")

    def stop_recording(self):
        """Stop recording and send audio to AI."""
        if self.is_listening:
            self.is_listening = False
            self.audio_handler.stop_recording()
            print("[RealtimeDialogueSystem] Stopped recording, sending to AI...")
            
            # Send commit event to process the audio
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.commit_audio_buffer(), 
                    self.loop
                )

    def end_conversation(self):
        """Properly end the conversation and clean up all resources."""
        print("[RealtimeDialogueSystem] Ending conversation...")
        self.active = False
        self.input_active = False
        
        # Stop any ongoing recording
        if self.is_listening:
            self.is_listening = False
            self.audio_handler.stop_recording()
        
        # Stop any AI speech
        if self.is_speaking:
            self.is_speaking = False
            self.audio_handler.stop_playback()
        
        # Clear buffers and text input
        self.audio_buffer = b''
        self.npc_message = ""
        self.user_input = ""
        self.conversation_history = []
        
        # *** CRITICAL FIX *** Close WebSocket connection
        if self.ws:
            print("[RealtimeDialogueSystem] Closing WebSocket connection...")
            if self.loop:
                # Schedule WebSocket closure in the event loop
                asyncio.run_coroutine_threadsafe(
                    self._close_websocket(), 
                    self.loop
                )
            self.ws = None
        
        # Reset conversation-specific state
        self.current_npc = None
        self.loop = None
        
        # Clean up audio handler (initialize new one for next conversation)
        self.audio_handler.cleanup()
        self.audio_handler = AudioHandler()  # Fresh audio handler for next conversation
        
        print("[RealtimeDialogueSystem] Conversation ended and cleaned up")
    
    async def _close_websocket(self):
        """Async helper to properly close WebSocket connection."""
        try:
            if self.ws and not self.ws.closed:
                await self.ws.close()
                print("[RealtimeDialogueSystem] WebSocket closed successfully")
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Error closing WebSocket: {e}")

    def interrupt_ai(self):
        """Interrupt the AI while it's speaking."""
        if self.is_speaking:
            print("[RealtimeDialogueSystem] Interrupting AI...")
            self.is_speaking = False
            
            # Stop any currently playing audio immediately
            self.audio_handler.stop_playback()
            
            # Clear any buffered audio
            self.audio_buffer = b''
            
            # Send cancel event to stop AI response
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.cancel_ai_response(), 
                    self.loop
                )

    def _stream_audio_continuously(self):
        """Stream audio chunks while recording."""
        while self.is_listening and self.audio_handler.is_recording:
            chunk = self.audio_handler.record_chunk()
            if chunk and self.loop:
                # Schedule the async send in the main event loop
                asyncio.run_coroutine_threadsafe(
                    self.send_audio_chunk(chunk), 
                    self.loop
                )
            time.sleep(0.01)  # Small delay to prevent overwhelming

    def render(self):
        if not self.active:
            return

        self.ui_surface.fill((0, 0, 0, 0))

        if self.active:
            box_height = 250  # Increased height for text input
            box_y = WINDOW_HEIGHT - box_height - 20
            
            # Make the background MUCH darker - almost black with some transparency
            box_color = (0, 0, 0, 230)  # Changed to very dark, mostly opaque background
            pygame.draw.rect(self.ui_surface, box_color, (20, box_y, WINDOW_WIDTH - 40, box_height))
            
            # White border
            pygame.draw.rect(self.ui_surface, (255, 255, 255, 255), (20, box_y, WINDOW_WIDTH - 40, box_height), 2)

            # Updated instructions for both voice and text
            npc_name = "Sarah (HR)" if self.current_npc == "HR" else "Michael (CEO)"
            instruction_text = f"Talking with {npc_name} | Ctrl+Space: record voice | Type & Enter: send text | Shift+Q: exit"
            if self.is_listening:
                instruction_text = f"🎤 RECORDING... | Press Ctrl+Space again to send to {npc_name} | Shift+Q: exit"
            elif self.is_speaking:
                voice_indicator = "👩‍💼" if self.current_npc == "HR" else "👨‍💼" 
                instruction_text = f"{voice_indicator} {npc_name} is speaking... | Ctrl+Space: interrupt | Type to interrupt | Shift+Q: exit"
                
            instruction_surface = self.font.render(instruction_text, True, (255, 255, 255))
            self.ui_surface.blit(instruction_surface, (40, box_y + 10))

            # NPC message in white
            if self.npc_message:
                self.render_text(self.ui_surface, self.npc_message, 40, box_y + 40)

            # Text input field
            if self.input_active:
                input_prompt = "> " + self.user_input + "_"
                input_surface = self.font.render(input_prompt, True, (255, 255, 255))
                self.ui_surface.blit(input_surface, (40, box_y + box_height - 40))

        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Setup for 2D rendering
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Bind and update texture
        glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw the UI texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()

        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def handle_input(self, event):
        print(f"[RealtimeDialogueSystem] handle_input called with event type: {event.type}, active: {self.active}")
        if not self.active:
            print("[RealtimeDialogueSystem] Not active, returning")
            return

        if event.type == pygame.KEYDOWN:
            print(f"[RealtimeDialogueSystem] Key down event: {pygame.key.name(event.key)}")
            keys = pygame.key.get_pressed()
            
            # Check for Shift+Q to exit chat
            if keys[pygame.K_LSHIFT] and event.key == pygame.K_q:
                self.end_conversation()  # Properly clean up resources
                print("[RealtimeDialogueSystem] Chat ended")
                # Return both the command and the initial position
                return {"command": "move_player_back", "position": self.initial_player_pos}

            # Handle Ctrl+Space for toggle recording (changed from just Space)
            if keys[pygame.K_LCTRL] and event.key == pygame.K_SPACE:
                print("[RealtimeDialogueSystem] CTRL+SPACE detected, calling toggle_recording!")
                self.toggle_recording()
                return  # Don't process other input when recording
            
            # Handle text input when input is active
            if self.input_active:
                if event.key == pygame.K_RETURN and self.user_input.strip():
                    # Send text message
                    self.send_text_message_sync()
                elif event.key == pygame.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                elif event.unicode.isprintable() and not keys[pygame.K_LCTRL]:
                    # Interrupt AI if it's speaking and user starts typing
                    if self.is_speaking and not self.user_input:
                        print("[RealtimeDialogueSystem] Interrupting AI for text input...")
                        self.interrupt_ai()
                    
                    # Add character to input (but not if Ctrl is held)
                    self.user_input += event.unicode

    def render_text(self, surface, text, x, y):
        max_width = WINDOW_WIDTH - 40
        line_height = 25
        
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        # Always use pure white with full opacity
        text_color = (255, 255, 255)
        
        for word in words:
            word_surface = self.font.render(word + ' ', True, text_color)
            word_width = word_surface.get_width()
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Render each line in pure white
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))  # Force white color
            surface.blit(text_surface, (x, y + i * line_height))
        
        return len(lines) * line_height

    async def connect_websocket(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        if not api_key:
            print("[RealtimeDialogueSystem] ERROR: No API key found in .env file")
            return
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
            "Content-Type": "application/json"
        }
        
        try:
            print(f"[RealtimeDialogueSystem] Attempting to connect to {self.url} with model {self.model}")
            print(f"[RealtimeDialogueSystem] Using API key: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")
            
            # Try different header parameter names for different websockets versions
            try:
                # For websockets >= 12.0
                self.ws = await websockets.connect(
                    f"{self.url}?model={self.model}",
                    additional_headers=headers,
                    ssl=self.ssl_context,
                    ping_interval=20,  # Keep connection alive
                    ping_timeout=20
                )
            except TypeError:
                # Fallback for older websockets versions
                self.ws = await websockets.connect(
                    f"{self.url}?model={self.model}",
                    extra_headers=headers,
                    ssl=self.ssl_context,
                    ping_interval=20,
                    ping_timeout=20
                )
            print("[RealtimeDialogueSystem] Connected to OpenAI Realtime API")
            
            # Configure session
            npc_voice = self.get_voice_for_npc()
            print(f"[RealtimeDialogueSystem] Using voice '{npc_voice}' for {self.current_npc}")
            
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": self.get_instructions_for_npc(),
                    "voice": npc_voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5
                    },
                    "temperature": 0.8
                }
            }
            print(f"[RealtimeDialogueSystem] Sending session config: {json.dumps(session_config, indent=2)}")
            await self.send_event(session_config)
            
            # Initiate conversation
            await self.send_event({"type": "response.create"})
            
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Failed to connect: {str(e)}")
            print(f"[RealtimeDialogueSystem] Error type: {type(e).__name__}")
            import traceback
            print(f"[RealtimeDialogueSystem] Traceback: {traceback.format_exc()}")
            self.ws = None

    async def send_event(self, event):
        """Send an event to the WebSocket server."""
        if self.ws:
            await self.ws.send(json.dumps(event))
            print(f"[RealtimeDialogueSystem] Sent event: {event['type']}")

    def get_instructions_for_npc(self):
        """Get system instructions based on current NPC."""
        base_prompt = """You are in a 3D virtual office environment engaging in voice conversation.
        Keep responses natural, conversational, and concise (2-3 sentences max).
        You can hear the user's voice and should respond naturally as if having a real conversation.
        Use appropriate pauses and natural speech patterns."""
        
        if self.current_npc == "HR":
            return f"""{base_prompt}
            You are Sarah Chen, HR Director at Venture Builder AI. You're warm, professional, 
            and helpful with employee matters. Speak naturally and offer practical assistance.
            Use a friendly, approachable tone while maintaining professionalism."""
        else:  # CEO
            return f"""{base_prompt}
            You are Michael Chen, CEO of Venture Builder AI. You're visionary yet approachable,
            passionate about venture building and AI. Share insights about the company and industry.
            Use a confident, engaging tone that inspires trust and enthusiasm."""

    def get_voice_for_npc(self):
        """Get appropriate voice based on current NPC."""
        if self.current_npc == "HR":
            return "alloy"  # Warm, professional female voice for HR Director Sarah Chen
        else:  # CEO
            return "echo"   # Confident, authoritative voice for CEO Michael Chen
            # Note: Changed from "nova" to "echo" - nova might be having issues

class World:
    def __init__(self):
        self.size = 5
        # Define office furniture colors
        self.colors = {
            'floor': (0.76, 0.6, 0.42),  # Light wood color
            'walls': (0.85, 0.85, 0.85),  # Changed to light gray (from 0.95)
            'desk': (0.6, 0.4, 0.2),  # Brown wood
            'chair': (0.2, 0.2, 0.2),  # Dark grey
            'computer': (0.1, 0.1, 0.1),  # Black
            'plant': (0.2, 0.5, 0.2),  # Green
            'partition': (0.3, 0.3, 0.3)  # Darker solid gray for booth walls
        }
        
    def draw_desk(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)  # Start at floor level
        glRotatef(rotation, 0, 1, 0)
        
        # Desk top (reduced size)
        glColor3f(*self.colors['desk'])
        glBegin(GL_QUADS)
        glVertex3f(-0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, 0.3)
        glVertex3f(-0.4, 0.4, 0.3)
        glEnd()
        
        # Desk legs (adjusted for new height)
        for x_offset, z_offset in [(-0.35, -0.25), (0.35, -0.25), (-0.35, 0.25), (0.35, 0.25)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.4, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.4, z_offset-0.02)
            glEnd()
        
        # Computer monitor (smaller)
        glColor3f(*self.colors['computer'])
        glTranslatef(-0.15, 0.4, 0)
        glBegin(GL_QUADS)
        glVertex3f(-0.1, 0, -0.05)
        glVertex3f(0.1, 0, -0.05)
        glVertex3f(0.1, 0.2, -0.05)
        glVertex3f(-0.1, 0.2, -0.05)
        glEnd()
        
        glPopMatrix()
    
    def draw_chair(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(rotation, 0, 1, 0)
        glColor3f(*self.colors['chair'])
        
        # Seat (lowered and smaller)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, 0.15)
        glVertex3f(-0.15, 0.25, 0.15)
        glEnd()
        
        # Back (adjusted height)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.5, -0.15)
        glVertex3f(-0.15, 0.5, -0.15)
        glEnd()
        
        # Chair legs (adjusted height)
        for x_offset, z_offset in [(-0.12, -0.12), (0.12, -0.12), (-0.12, 0.12), (0.12, 0.12)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.25, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.25, z_offset-0.02)
            glEnd()
            
        glPopMatrix()
    
    def draw_plant(self, x, z):
        glPushMatrix()
        glTranslatef(x, 0, z)
        
        # Plant pot (smaller)
        glColor3f(0.4, 0.2, 0.1)  # Brown pot
        pot_radius = 0.1
        pot_height = 0.15
        segments = 8
        
        # Draw the pot sides
        glBegin(GL_QUADS)
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            x1 = math.cos(angle1) * pot_radius
            z1 = math.sin(angle1) * pot_radius
            x2 = math.cos(angle2) * pot_radius
            z2 = math.sin(angle2) * pot_radius
            glVertex3f(x1, 0, z1)
            glVertex3f(x2, 0, z2)
            glVertex3f(x2, pot_height, z2)
            glVertex3f(x1, pot_height, z1)
        glEnd()
        
        # Plant leaves (smaller)
        glColor3f(*self.colors['plant'])
        glTranslatef(0, pot_height, 0)
        leaf_size = 0.15
        num_leaves = 6
        for i in range(num_leaves):
            angle = (i / num_leaves) * 2 * math.pi
            x = math.cos(angle) * leaf_size
            z = math.sin(angle) * leaf_size
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(x, leaf_size, z)
            glVertex3f(z, leaf_size/2, -x)
            glEnd()
        
        glPopMatrix()
        
    def draw(self):
        # Set material properties
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Draw floor at Y=0
        glBegin(GL_QUADS)
        glColor3f(*self.colors['floor'])
        glNormal3f(0, 1, 0)
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 0, -self.size)
        glEnd()
        
        # Draw walls starting from floor level
        glBegin(GL_QUADS)
        glColor3f(*self.colors['walls'])
        
        # Front wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 2, -self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Back wall
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(-self.size, 2, self.size)
        
        # Left wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(-self.size, 2, self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Right wall
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(self.size, 2, -self.size)
        glEnd()
        
        # Draw office furniture in a more realistic arrangement
        # HR Area (left side)
        self.draw_desk(-4, -2, 90)
        self.draw_chair(-3.5, -2, 90)
        self.draw_partition_walls(-4, -2)  # Add booth walls for HR
        
        # CEO Area (right side)
        self.draw_desk(4, 1, -90)
        self.draw_chair(3.5, 1, -90)
        self.draw_partition_walls(4, 1)  # Add booth walls for CEO
        
        # Plants in corners (moved closer to walls)
        self.draw_plant(-4.5, -4.5)
        self.draw_plant(4.5, -4.5)
        self.draw_plant(-4.5, 4.5)
        self.draw_plant(4.5, 4.5)

    def draw_partition_walls(self, x, z):
        """Draw booth partition walls - all surfaces in solid gray"""
        glColor3f(0.3, 0.3, 0.3)  # Solid gray for all walls
        
        # Back wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z)
        glScalef(0.05, 1.0, 1.0)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()
        
        # Side wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z + 0.5)  # Moved closer
        glRotatef(90, 0, 1, 0)
        glScalef(0.05, 1.0, 0.8)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()

class Player:
    def __init__(self):
        self.pos = [0, 0.5, 0]  # Lowered Y position to be just above floor
        self.rot = [0, 0, 0]
        self.speed = 0.3
        self.mouse_sensitivity = 0.5
        
    def move(self, dx, dz):
        # Convert rotation to radians (negative because OpenGL uses clockwise rotation)
        angle = math.radians(-self.rot[1])
        
        # Calculate movement vector
        move_x = (dx * math.cos(angle) + dz * math.sin(angle)) * self.speed
        move_z = (-dx * math.sin(angle) + dz * math.cos(angle)) * self.speed
        
        # Calculate new position
        new_x = self.pos[0] + move_x
        new_z = self.pos[2] + move_z
        
        # Wall collision check (room is 10x10)
        room_limit = 4.5  # Slightly less than room size/2 to prevent wall clipping
        if abs(new_x) < room_limit:
            self.pos[0] = new_x
        if abs(new_z) < room_limit:
            self.pos[2] = new_z

    def update_rotation(self, dx, dy):
        # Multiply mouse movement by sensitivity for faster turning
        self.rot[1] += dx * self.mouse_sensitivity

class NPC:
    def __init__(self, x, y, z, role="HR"):
        self.scale = 0.6  # Make NPCs smaller (about 60% of current size)
        # Position them beside the desks, at ground level
        # Adjust Y position to be half their height (accounting for scale)
        self.pos = [x, 0.65, z]  # This puts their feet on the ground
        self.size = 0.5
        self.role = role
        
        # Enhanced color palette
        self.skin_color = (0.8, 0.7, 0.6)  # Neutral skin tone
        self.hair_color = (0.2, 0.15, 0.1) if role == "HR" else (0.3, 0.3, 0.3)  # Dark brown vs gray
        
        # Updated clothing colors
        if role == "HR":
            self.clothes_primary = (0.8, 0.2, 0.2)    # Bright red
            self.clothes_secondary = (0.6, 0.15, 0.15) # Darker red
        else:  # CEO
            self.clothes_primary = (0.2, 0.3, 0.8)    # Bright blue
            self.clothes_secondary = (0.15, 0.2, 0.6)  # Darker blue

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glScalef(self.scale, self.scale, self.scale)
        
        # Head
        glColor3f(*self.skin_color)
        draw_sphere(0.12, 16, 16)
        
        # Hair (slightly larger than head)
        glColor3f(*self.hair_color)
        glPushMatrix()
        glTranslatef(0, 0.05, 0)  # Slightly above head
        draw_sphere(0.13, 16, 16)
        glPopMatrix()
        
        # Body (torso)
        glColor3f(*self.clothes_primary)
        glPushMatrix()
        glTranslatef(0, -0.3, 0)  # Move down from head
        glScalef(0.3, 0.4, 0.2)   # Make it rectangular
        draw_cube()
        glPopMatrix()
        
        # Arms
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.2, 0.2]:  # Left and right arms
            glPushMatrix()
            glTranslatef(x_offset, -0.3, 0)
            glScalef(0.1, 0.4, 0.1)
            draw_cube()
            glPopMatrix()
        
        # Legs
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.1, 0.1]:  # Left and right legs
            glPushMatrix()
            glTranslatef(x_offset, -0.8, 0)
            glScalef(0.1, 0.5, 0.1)
            draw_cube()
            glPopMatrix()
        
        glPopMatrix()

class MenuScreen:
    def __init__(self):
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.active = True
        self.start_time = time.time()
        
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Create a surface for 2D rendering
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        
        # Calculate vertical positions
        center_y = WINDOW_HEIGHT // 2
        title_y = center_y - 100
        subtitle_y = center_y - 20
        prompt_y = center_y + 100
        
        # Render title with "typing" effect
        elapsed_time = time.time() - self.start_time
        title_chars = int(min(len(TITLE), elapsed_time * 15))  # Type 15 chars per second
        partial_title = TITLE[:title_chars]
        title_surface = self.font_large.render(partial_title, True, MENU_TEXT_COLOR)
        title_x = (WINDOW_WIDTH - title_surface.get_width()) // 2
        surface.blit(title_surface, (title_x, title_y))
        
        # Render subtitle with fade-in effect
        if elapsed_time > len(TITLE) / 15:  # Start after title is typed
            subtitle_alpha = min(255, int((elapsed_time - len(TITLE) / 15) * 255))
            subtitle_surface = self.font_medium.render(SUBTITLE, True, MENU_TEXT_COLOR)
            subtitle_surface.set_alpha(subtitle_alpha)
            subtitle_x = (WINDOW_WIDTH - subtitle_surface.get_width()) // 2
            surface.blit(subtitle_surface, (subtitle_x, subtitle_y))
        
        # Render "Press ENTER" with blinking effect
        if elapsed_time > (len(TITLE) / 15 + 1):  # Start after subtitle fade
            if int(elapsed_time * 2) % 2:  # Blink every 0.5 seconds
                prompt_text = "Press ENTER to start"
                prompt_surface = self.font_small.render(prompt_text, True, MENU_TEXT_COLOR)
                prompt_x = (WINDOW_WIDTH - prompt_surface.get_width()) // 2
                surface.blit(prompt_surface, (prompt_x, prompt_y))
        
        # Add some retro effects (scanlines)
        for y in range(0, WINDOW_HEIGHT, 4):
            pygame.draw.line(surface, (0, 50, 0), (0, y), (WINDOW_WIDTH, y))
        
        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        
        # Set up orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Render the texture in OpenGL
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Draw the texture
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 0); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        # Reset OpenGL state for 3D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

# Modify the Game3D class to include the menu
class Game3D:
    def __init__(self):
        self.menu = MenuScreen()
        self.player = Player()
        self.world = World()
        self.dialogue = RealtimeDialogueSystem()
        self.hr_npc = NPC(-3.3, 0, -2, "HR")  # Moved beside the desk
        self.ceo_npc = NPC(3.3, 0, 1, "CEO")  # Moved beside the desk
        self.interaction_distance = 2.0
        self.last_interaction_time = 0

    def move_player_away_from_npc(self, npc_pos):
        # Calculate direction vector from NPC to player
        dx = self.player.pos[0] - npc_pos[0]
        dz = self.player.pos[2] - npc_pos[2]
        
        # Normalize the vector
        distance = math.sqrt(dx*dx + dz*dz)
        if distance > 0:
            dx /= distance
            dz /= distance
        
        # Move player back by 3 units
        self.player.pos[0] = npc_pos[0] + (dx * 3)
        self.player.pos[2] = npc_pos[2] + (dz * 3)
    


    def run(self):
        running = True
        while running:
            if self.menu.active:
                # Menu loop
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN and time.time() - self.menu.start_time > (len(TITLE) / 15 + 1):
                            self.menu.active = False
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                
                self.menu.render()
            else:
                # Main game loop
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.mouse.set_visible(True)
                            pygame.event.set_grab(False)
                            running = False
                        
                        # Handle dialogue input and check for exit command
                        if self.dialogue.active:
                            result = self.dialogue.handle_input(event)
                            if isinstance(result, dict) and result.get("command") == "move_player_back":
                                # Move player away from the current NPC
                                current_npc = self.hr_npc if self.dialogue.current_npc == "HR" else self.ceo_npc
                                self.move_player_away_from_npc(current_npc.pos)
                                
                    elif event.type == pygame.MOUSEMOTION:
                        x, y = event.rel
                        self.player.update_rotation(x, y)

                # Handle keyboard input for movement (keep this blocked during dialogue)
                if not self.dialogue.active:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_w]: self.player.move(0, -1)
                    if keys[pygame.K_s]: self.player.move(0, 1)
                    if keys[pygame.K_a]: self.player.move(-1, 0)
                    if keys[pygame.K_d]: self.player.move(1, 0)

                # Check NPC interactions
                current_time = time.time()
                if current_time - self.last_interaction_time > 0.5:  # Cooldown on interactions
                    # Check distance to HR NPC
                    dx = self.player.pos[0] - self.hr_npc.pos[0]
                    dz = self.player.pos[2] - self.hr_npc.pos[2]
                    hr_distance = math.sqrt(dx*dx + dz*dz)
                    
                    # Check distance to CEO NPC
                    dx = self.player.pos[0] - self.ceo_npc.pos[0]
                    dz = self.player.pos[2] - self.ceo_npc.pos[2]
                    ceo_distance = math.sqrt(dx*dx + dz*dz)
                    
                    if hr_distance < self.interaction_distance and not self.dialogue.active:
                        self.dialogue.start_conversation("HR", self.player.pos)
                        self.last_interaction_time = current_time
                    elif ceo_distance < self.interaction_distance and not self.dialogue.active:
                        self.dialogue.start_conversation("CEO", self.player.pos)
                        self.last_interaction_time = current_time

                # Clear the screen and depth buffer
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Save the current matrix
                glPushMatrix()

                # Apply player rotation and position
                glRotatef(self.player.rot[0], 1, 0, 0)
                glRotatef(self.player.rot[1], 0, 1, 0)
                glTranslatef(-self.player.pos[0], -self.player.pos[1], -self.player.pos[2])

                # Draw the world and NPCs
                self.world.draw()
                self.hr_npc.draw()
                self.ceo_npc.draw()

                # Restore the matrix
                glPopMatrix()

                # Render dialogue system (if active)
                self.dialogue.render()

                # Swap the buffers
                pygame.display.flip()

                # Maintain 60 FPS
                pygame.time.Clock().tick(60)

        pygame.quit()

# Create and run game
game = Game3D()
game.run()

