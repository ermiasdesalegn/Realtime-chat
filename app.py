# Adventure Game
import os
# Set appropriate video driver for the platform
import platform
if platform.system() == 'Darwin':  # macOS
    os.environ['SDL_VIDEODRIVER'] = 'cocoa'
# Windows will use the default video driver
# REMOVED: os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Unhide pygame support info

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
from datetime import datetime
from pydub import AudioSegment
import io
import wave
import traceback

# Helper function for real-time terminal output
def debug_print(message, category="DEBUG"):
    """Print debug messages with timestamps and immediate flushing"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    formatted_msg = f"[{timestamp}] [{category}] {message}"
    print(formatted_msg)
    sys.stdout.flush()  # Ensure immediate output

# Load environment variables
load_dotenv()
# Ensure OpenAI API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    debug_print("API key not found. Please set OPENAI_API_KEY in your .env file.", "ERROR")
    sys.exit(1)
client = OpenAI(api_key=api_key)
debug_print("API key loaded successfully.", "OpenAI")

# Initialize Pygame
debug_print("Initializing Pygame...", "INIT")
pygame.init()
display = (800, 600)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
screen = pygame.display.get_surface()
debug_print("Pygame initialized successfully.", "INIT")

# Set up the camera and perspective
debug_print("Setting up OpenGL...", "INIT")
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
debug_print("OpenGL setup complete.", "INIT")

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
        debug_print("Initializing AudioHandler...", "AUDIO")
        self.audio_lock = threading.Lock()  # Add lock for thread safety
        self.p = None
        self.stream = None
        self.audio_buffer = b''
        self.chunk_size = 256  # REDUCED from 512 for even better performance and lower crash probability
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
        debug_print("Attempting to initialize PyAudio...", "AUDIO")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if self.p is not None:
                    try:
                        self.p.terminate()
                        debug_print("Terminated existing PyAudio instance", "AUDIO")
                    except:
                        pass
                
                self.p = pyaudio.PyAudio()
                self._initialized = True
                debug_print("✅ PyAudio initialized successfully!", "AUDIO")
                
                # Print available audio devices with real-time formatting
                debug_print("🔍 Scanning available audio devices...", "AUDIO")
                device_count = self.p.get_device_count()
                debug_print(f"Found {device_count} audio devices:", "AUDIO")
                
                for i in range(device_count):
                    try:
                        dev_info = self.p.get_device_info_by_index(i)
                        device_type = "🎤INPUT" if dev_info['maxInputChannels'] > 0 else "🔊OUTPUT"
                        if dev_info['maxInputChannels'] > 0 and dev_info['maxOutputChannels'] > 0:
                            device_type = "🎧BOTH"
                        debug_print(f"  [{i:2d}] {device_type} | {dev_info['name'][:50]:<50} | In:{dev_info['maxInputChannels']} Out:{dev_info['maxOutputChannels']}", "AUDIO")
                    except Exception as e:
                        debug_print(f"  [{i:2d}] ERROR reading device info: {e}", "AUDIO")
                
                return
            except Exception as e:
                debug_print(f"Audio init attempt {attempt+1} failed: {e}", "AUDIO_ERROR")
                time.sleep(1)
        debug_print("Failed to initialize audio after 3 attempts", "CRITICAL")
        self._initialized = False
        


    def start_recording(self):
        """Start recording audio from microphone."""
        debug_print("🎤 STARTING RECORDING SESSION...", "AUDIO_REC")
        self.is_recording = True
        self.audio_buffer = b''
        
        # Ensure PyAudio is initialized
        if not self._initialized or self.p is None:
            debug_print("⚠️ PyAudio not initialized, attempting to reinitialize...", "AUDIO_REC")
            self._initialize_pyaudio()
            
        if not self._initialized:
            debug_print("❌ CRITICAL ERROR: Failed to initialize PyAudio!", "AUDIO_ERROR")
            self.is_recording = False
            return
        
        # List all available input devices with real-time status
        input_devices = []
        debug_print("🔍 Scanning for input devices...", "AUDIO_REC")
        try:
            for i in range(self.p.get_device_count()):
                try:
                    dev_info = self.p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        input_devices.append((i, dev_info))
                        debug_print(f"  ✅ Found input device [{i}]: {dev_info['name'][:40]} (channels: {dev_info['maxInputChannels']})", "AUDIO_REC")
                except:
                    continue
        except Exception as e:
            debug_print(f"❌ Error scanning device list: {e}", "AUDIO_ERROR")
            self.is_recording = False
            return
        
        if not input_devices:
            debug_print("❌ CRITICAL: No input devices found! Check microphone connection.", "AUDIO_ERROR")
            self.is_recording = False
            return
        
        # Try devices in order: default first, then others
        devices_to_try = []
        
        # Try to get default device first
        debug_print("🎯 Attempting to use default input device...", "AUDIO_REC")
        try:
            default_input = self.p.get_default_input_device_info()
            devices_to_try.append((default_input['index'], default_input, "default"))
            debug_print(f"✅ Default input device: {default_input['name'][:40]}", "AUDIO_REC")
        except Exception as e:
            debug_print(f"⚠️ No default input device: {e}", "AUDIO_REC")
        
        # Add all other input devices
        for device_index, device_info in input_devices:
            if not any(d[0] == device_index for d in devices_to_try):
                devices_to_try.append((device_index, device_info, "fallback"))
        
        # Try each device until one works
        debug_print(f"🔄 Testing {len(devices_to_try)} devices...", "AUDIO_REC")
        for device_index, device_info, device_type in devices_to_try:
            debug_print(f"🧪 Testing {device_type.upper()} device [{device_index}]: {device_info['name'][:30]}...", "AUDIO_REC")
            try:
                # Try different sample rates if the default doesn't work
                rates_to_try = [self.rate, 44100, 48000, 22050, 16000]
                
                for rate in rates_to_try:
                    debug_print(f"   📡 Trying {rate}Hz...", "AUDIO_REC")
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
                        test_data = self.stream.read(64, exception_on_overflow=False)
                        
                        debug_print(f"🎉 SUCCESS! Recording active on device [{device_index}] at {rate}Hz", "AUDIO_SUCCESS")
                        debug_print(f"   📊 Stream config: {self.channels}ch, {self.chunk_size} buffer", "AUDIO_SUCCESS")
                        self.rate = rate  # Update rate to the working one
                        return
                        
                    except Exception as rate_error:
                        debug_print(f"   ❌ {rate}Hz failed: {str(rate_error)[:50]}", "AUDIO_REC")
                        if self.stream:
                            try:
                                self.stream.close()
                            except:
                                pass
                            self.stream = None
                        continue
                        
            except Exception as device_error:
                debug_print(f"❌ Device [{device_index}] failed: {str(device_error)[:50]}", "AUDIO_REC")
                continue
        
        # If we get here, no device worked
        debug_print("🚨 CRITICAL FAILURE: Could not start recording with ANY device!", "AUDIO_ERROR")
        debug_print("📋 TROUBLESHOOTING CHECKLIST:", "AUDIO_ERROR")
        debug_print("  1. 🎤 Microphone is connected and powered on", "AUDIO_ERROR")
        debug_print("  2. 🔒 Windows Privacy Settings allow microphone access", "AUDIO_ERROR")
        debug_print("  3. 🚫 No other applications are using the microphone", "AUDIO_ERROR")
        debug_print("  4. 🔧 Audio drivers are up to date", "AUDIO_ERROR")
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
        
    def play_audio(self, audio_data, dialogue_system=None):
        """Play audio data."""
        debug_print(f"🎧 AudioHandler.play_audio called with {len(audio_data)} bytes", "AUDIO_PLAYBACK")
        self.dialogue_system = dialogue_system  # Store reference to update is_speaking

        # If already playing, just queue and return
        if self.is_playing:
            debug_print("⏸️ Audio is already playing, queuing new response", "AUDIO_PLAYBACK")
            if hasattr(self.dialogue_system, 'audio_queue'):
                self.dialogue_system.audio_queue.append(audio_data)
            return

        # Clear the stop event for new playback
        self._stop_event.clear()

        def play():
            local_stream = None
            try:
                debug_print("🎵 Starting audio playback thread...", "AUDIO_PLAYBACK")
                self.is_playing = True
                debug_print(f"🔧 Opening audio output stream: {self.format}, {self.channels}ch, {self.rate}Hz", "AUDIO_PLAYBACK")
                local_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    output=True,
                    frames_per_buffer=128  # REDUCED from 256 for maximum performance and instant interruption
                )
                self.playback_stream = local_stream
                debug_print("✅ Audio output stream opened successfully", "AUDIO_PLAYBACK")
                chunk_size = 128  # REDUCED from 256 for maximum performance and instant interruption
                total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
                chunks_played = 0
                debug_print(f"▶️ Playing {total_chunks} audio chunks...", "AUDIO_PLAYBACK")
                for i in range(0, len(audio_data), chunk_size):
                    # Check for stop signals
                    if self._stop_event.is_set():
                        debug_print("⏸️ Audio playback interrupted", "AUDIO_PLAYBACK")
                        break
                    if not self.is_playing:
                        debug_print("⏹️ Audio playback stopped", "AUDIO_PLAYBACK")
                        break
                    chunk = audio_data[i:i+chunk_size]
                    try:
                        local_stream.write(chunk)
                        chunks_played += 1
                        if chunks_played % 50 == 0:  # REDUCED frequency of progress updates to lower CPU usage
                            debug_print(f"🎵 Progress: {chunks_played}/{total_chunks} chunks played", "AUDIO_PROGRESS")
                    except Exception as e:
                        debug_print(f"❌ Audio chunk write error: {e}", "AUDIO_ERROR")
                        break
                try:
                    if local_stream:
                        local_stream.stop_stream()
                        import time
                        time.sleep(0.1)
                        debug_print(f"🎶 Audio playback completed - {chunks_played}/{total_chunks} chunks played", "AUDIO_PLAYBACK")
                except:
                    pass
            except Exception as e:
                debug_print(f"❌ Audio system error during playback: {e}", "AUDIO_ERROR")
            finally:
                if local_stream:
                    try:
                        if not hasattr(local_stream, '_stopped') or not local_stream._stopped:
                            local_stream.stop_stream()
                            local_stream.close()
                            debug_print("🔇 Audio output stream closed", "AUDIO_PLAYBACK")
                    except:
                        pass
                self.playback_stream = None
                self.is_playing = False
                if hasattr(self, 'dialogue_system') and self.dialogue_system:
                    self.dialogue_system.is_speaking = False
                    self.dialogue_system.ai_speaking_pause = False
                    import time
                    self.dialogue_system.last_response_time = time.time()
                    # Add this line:
                    self.dialogue_system.last_ai_speaking_end = time.time()
                    debug_print("✅ AI speaking status set to False - Ready for user input with cooldown", "AUDIO_PLAYBACK")
                    self.dialogue_system.audio_playing = False
                    # Only play next queued audio if not interrupting and not in user speech
                    if (
                        hasattr(self.dialogue_system, 'audio_queue') and self.dialogue_system.audio_queue
                        and not self.dialogue_system.interrupting
                        and not self.dialogue_system.is_user_speaking
                        and len(self.dialogue_system.audio_queue) <= 3  # LIMIT queue size to prevent memory buildup
                    ):
                        next_audio = self.dialogue_system.audio_queue.pop(0)
                        debug_print(f"▶️ Playing next queued AI audio response ({len(self.dialogue_system.audio_queue)} remaining)", "AUDIO_PLAYBACK")
                        self.dialogue_system.audio_playing = True
                        self.dialogue_system.audio_handler.play_audio(next_audio, dialogue_system=self.dialogue_system)
                    elif hasattr(self.dialogue_system, 'audio_queue') and len(self.dialogue_system.audio_queue) > 3:
                        # Clear excessive queue to prevent memory issues
                        queue_size = len(self.dialogue_system.audio_queue)
                        self.dialogue_system.audio_queue.clear()
                        debug_print(f"🗑️ Cleared excessive audio queue ({queue_size} items) to prevent memory issues", "AUDIO_CLEANUP")
                        # Force garbage collection
                        import gc
                        gc.collect()
        # Wait for any existing thread to finish (only if not already playing)
        import threading
        if self.playback_thread and self.playback_thread.is_alive():
            if threading.current_thread() != self.playback_thread:
                self._stop_event.set()
                self.playback_thread.join(timeout=2.0)
        debug_print("🧵 Starting audio playback thread...", "AUDIO_PLAYBACK")
        self.playback_thread = threading.Thread(target=play)
        self.playback_thread.daemon = False
        self.playback_thread.start()

    def stop_playback(self):
        """Stop any currently playing audio."""
        if self.is_playing:
            debug_print("[AUDIO] stop_playback called: Stopping current audio playback.", "AUDIO_DEBUG")
            self.is_playing = False
            self._stop_event.set()  # Signal the playback thread to stop
            # IMMEDIATELY stop and close the stream for instant interruption
            if self.playback_stream:
                try:
                    self.playback_stream.stop_stream()
                    self.playback_stream.close()
                except:
                    pass
                self.playback_stream = None
            # Wait for the thread to finish
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=3.0)
        else:
            debug_print("[AUDIO] stop_playback called: No audio was playing.", "AUDIO_DEBUG")

    def cleanup(self):
        """Clean up resources."""
        try:
            print("[AudioHandler] Cleaning up...")
            
            # Stop recording with error handling
            try:
                if self.stream:
                    self.stop_recording()
            except Exception as e:
                print(f"[AudioHandler] Error stopping recording: {e}")
            
            # Stop playback with error handling
            try:
                if self.is_playing and self.playback_thread and self.playback_thread.is_alive():
                    print("[AudioHandler] Waiting for audio playback to complete...")
                    import threading
                    if threading.current_thread() != self.playback_thread:
                        self.playback_thread.join(timeout=3.0)  # Reduced timeout
                        
                self.stop_playback()
            except Exception as e:
                print(f"[AudioHandler] Error stopping playback: {e}")
            
            # Force terminate playback thread if still alive
            try:
                if self.playback_thread and self.playback_thread.is_alive():
                    self._stop_event.set()
                    import threading
                    if threading.current_thread() != self.playback_thread:
                        self.playback_thread.join(timeout=2.0)  # Shorter timeout
                        if self.playback_thread.is_alive():
                            print("[AudioHandler] WARNING: Playback thread did not terminate gracefully")
            except Exception as e:
                print(f"[AudioHandler] Error terminating playback thread: {e}")
            
            # Terminate PyAudio with error handling
            try:
                if self.p:
                    self.p.terminate()
                    print("[AudioHandler] PyAudio terminated successfully")
            except Exception as e:
                print(f"[AudioHandler] Error during PyAudio cleanup: {e}")
            
            print("[AudioHandler] Cleanup complete")
            
        except Exception as e:
            print(f"[AudioHandler] CRITICAL ERROR during cleanup: {e}")
            # Emergency cleanup
            try:
                self.is_recording = False
                self.is_playing = False
                if hasattr(self, '_stop_event'):
                    self._stop_event.set()
            except:
                pass

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
            x = math.cos(lng) * radius
            y = math.sin(lng) * radius
            
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
        
        # Continuous voice input controls
        self.is_listening = False
        self.loop = None  # Will store the event loop
        self.continuous_recording = True  # Always recording for seamless interaction
        self.voice_activity_threshold = 0.22  # INCREASED from 0.15 for higher speech power requirement
        self.noise_floor = 0.10  # INCREASED from 0.08 to filter out more background noise
        self.max_silence_duration = 0.3  # Shorter silence before stopping speech
        self.min_speech_duration = 0.1  # Minimal duration to consider as intentional speech
        self.is_user_speaking = False  # Track if user is currently speaking
        self.last_audio_level = 0.0  # Track audio levels for VAD
        self.current_response_id = None  # Track current response to prevent overlaps
        self.response_in_progress = False  # Flag to prevent multiple responses
        self.ai_speaking_pause = False  # Prevent interruptions when AI is actively speaking
        self.speech_confirmation_buffer = []  # Buffer to confirm sustained speech
        
        # Text input controls
        self.user_input = ""
        self.input_active = False
        self.conversation_history = []  # For text-based conversations
        
        # Mode switching: voice vs text
        self.text_mode_active = False  # When True, disable voice interaction
        
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

        # Add an audio queue and a flag for playback
        self.audio_queue = []  # Add to __init__
        self.audio_playing = False  # Add to __init__
        self.ai_response_lock = threading.Lock()  # Prevent overlapping responses
        self.speech_start_time = None  # Track when user speech starts
        self.voice_frequency_range = (150, 3000)  # Tighter human voice frequency range
        self.interrupting = False  # Add this flag for instant interruption feedback
        self.last_ai_speaking_end = 0  # For VAD cooldown after AI speech
        self.waiting_for_user_input = False  # Prevent repeated nudges/AI self-talking

    def start_conversation(self, npc_role="HR", player_pos=None):
        # UI state: open dialogue box immediately
        debug_print(f"🔄 START_CONVERSATION called for {npc_role}", "DIALOGUE_START")
        self.active = True
        self.input_active = True
        self.current_npc = npc_role
        debug_print(f"✅ Dialogue active flag set: {self.active}", "DIALOGUE_START")
        
        # Always use the greeting from AI_AGENTS
        agent = AI_AGENTS.get(npc_role, {})
        greeting = agent.get("greeting", "")
        self.npc_message = greeting
        self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
        debug_print(f"📝 Greeting set: {greeting[:50]}...", "DIALOGUE_START")
        
        # Now do the rest (debug_print, background thread, etc.)
        debug_print(f"🗣️ STARTING SEAMLESS CONVERSATION with {npc_role}", "DIALOGUE")
        debug_print(f"📍 Player position: {player_pos}", "DIALOGUE")
        try:
            debug_print("⚙️ Setting up seamless conversation environment...", "DIALOGUE")
            # Reset voice activity detection state
            self.is_user_speaking = False
            self.silence_duration = 0
            self.last_audio_level = 0.0
            self.last_response_time = 0  # Reset cooldown timer
            self.conversation_history = [{
                "role": "system",
                "content": self.get_instructions_for_npc()
            }]
            debug_print(f"💬 Initial message set: {self.npc_message[:50]}...", "DIALOGUE")
            
            # Play greeting immediately
            debug_print(f"🎵 Starting TTS for greeting...", "DIALOGUE_START")
            self.text_to_speech(greeting)
            debug_print(f"✅ TTS call completed", "DIALOGUE_START")
            
            # Do NOT ask the AI model to generate its own greeting for the first message
            # Show conversation participants
            speaker_name = "Sarah (HR Director)" if npc_role == "HR" else "Michael (CEO)"
            debug_print("=" * 60, "CONVERSATION")
            debug_print(f"🎭 STARTING SEAMLESS CONVERSATION WITH {speaker_name.upper()}", "CONVERSATION")
            debug_print(f"👥 Participants: User ↔️ {speaker_name}", "CONVERSATION")
            debug_print(f"🎙️ Mode: CONTINUOUS VOICE (No buttons required)", "CONVERSATION")
            debug_print("=" * 60, "CONVERSATION")
            debug_print(f"💬 {speaker_name} says: \"{self.npc_message}\"", "SPEAKER")
            debug_print(f"🔑 API key status: {'✅ Present' if api_key else '❌ Missing'}", "DIALOGUE")
            debug_print(f"🔊 Audio handler status: {'✅ Ready' if self.audio_handler else '❌ Not initialized'}", "DIALOGUE")
            debug_print(f"📊 Dialogue state: active={self.active}, continuous_recording={self.continuous_recording}", "DIALOGUE")
            
            # Start the async conversation in a separate thread
            debug_print("🧵 Creating seamless conversation thread...", "DIALOGUE")
            self.conversation_thread = threading.Thread(target=self._run_async_conversation)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()
            debug_print("✅ Seamless conversation thread started successfully!", "DIALOGUE")
        except Exception as e:
            debug_print(f"❌ CRITICAL ERROR starting conversation: {str(e)}", "DIALOGUE_ERROR")
            debug_print(f"🔍 Error type: {type(e).__name__}", "DIALOGUE_ERROR")
            import traceback
            debug_print(f"📋 Traceback: {traceback.format_exc()}", "DIALOGUE_ERROR")
            self.active = False

    def _run_async_conversation(self):
        """Run the async conversation loop in a separate thread."""
        try:
            asyncio.run(self._conversation_loop())
        except Exception as e:
            print(f"[RealtimeDialogueSystem] Conversation error: {e}")
            self.active = False

    async def _conversation_loop(self):
        """Main async conversation loop with continuous voice monitoring."""
        self.loop = asyncio.get_running_loop()
        await self.connect_websocket()
        if not self.ws:
            debug_print("❌ Failed to connect WebSocket, ending conversation", "CONVERSATION_ERROR")
            self.active = False
            return
        debug_print("🎤 Starting continuous voice monitoring...", "CONTINUOUS_VOICE")
        self.start_continuous_recording()
        listen_task = asyncio.create_task(self._listen_for_events())
        voice_task = asyncio.create_task(self._continuous_voice_monitoring())
        nudge_task = asyncio.create_task(self._user_silence_nudge())
        try:
            await asyncio.gather(listen_task, voice_task, nudge_task)
        except Exception as e:
            debug_print(f"❌ Error in conversation loop: {e}", "CONVERSATION_ERROR")
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
        import datetime
        event_type = event.get("type")
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_print(f"📨 Received event: {event_type} at {now}", "WS_EVENT")
        if event_type == "error":
            debug_print(f"❌ API ERROR: {event['error']['message']}", "WS_ERROR")
        elif event_type == "response.audio.delta":
            # Accept audio data and accumulate it
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
            debug_print(f"🔊 [AUDIO_DELTA] {now} - Received {len(audio_data)} bytes, buffer now {len(self.audio_buffer)} bytes", "AUDIO_DEBUG")
        elif event_type == "response.audio.done":
            debug_print(f"🔊 [AUDIO_DONE] {now} - Final buffer size: {len(self.audio_buffer)} bytes", "AUDIO_DEBUG")
            if self.audio_buffer:
                debug_print(f"🎵 Playing AI audio response ({len(self.audio_buffer)} bytes)", "AUDIO_PLAYBACK")
                # Instead of stopping previous audio, queue new audio if already playing
                if self.audio_handler.is_playing or self.audio_playing:
                    # Check queue size to prevent memory buildup
                    if len(self.audio_queue) < 3:  # LIMIT queue size
                        debug_print("⏸️ Audio is already playing, queuing new response", "AUDIO_PLAYBACK")
                        self.audio_queue.append(self.audio_buffer)
                    else:
                        debug_print("⚠️ Audio queue full, dropping oldest response to prevent memory issues", "AUDIO_QUEUE_FULL")
                        self.audio_queue.pop(0)  # Remove oldest
                        self.audio_queue.append(self.audio_buffer)  # Add newest
                        # Force garbage collection
                        import gc
                        gc.collect()
                else:
                    self.is_speaking = True
                    self.ai_speaking_pause = True
                    debug_print("🔒 AI SPEAKING - Protected from interruption for sustained speech", "AUDIO_PROTECT")
                    self.speech_confirmation_buffer = []
                    self.is_user_speaking = False
                    self.audio_playing = True
                self.audio_handler.play_audio(self.audio_buffer, dialogue_system=self)
                self.audio_buffer = b''
            else:
                debug_print("⚠️ No audio data to play", "AUDIO_PLAYBACK")
                self.is_speaking = False
                self.ai_speaking_pause = False
        elif event_type == "response.text.delta":
            # Build text in larger chunks for faster display (reduced lag)
            if "delta" in event:
                self.npc_message += event["delta"]  # Update dialogue box in real time as AI speaks
                debug_print(f"⚡ INSTANT Text chunk: '{event['delta']}'", "TEXT_INSTANT")
                # Only show progress every few characters to reduce terminal spam
                if len(event["delta"]) > 3 or event["delta"].endswith(' '):
                    debug_print(f"📄 Building message: '{self.npc_message[-50:]}'", "TEXT_BUILD")
        elif event_type == "response.text.done":
            debug_print(f"✅ FINAL TEXT READY: {self.npc_message}", "TEXT_FINAL")
            
            # Show who is speaking in the terminal (for text-based responses)
            if self.npc_message and not self.is_speaking:  # Only if not already shown by voice transcript
                speaker_name = "Sarah (HR Director)" if self.current_npc == "HR" else "Michael (CEO)"
                debug_print(f"💬 {speaker_name} says: \"{self.npc_message}\"", "SPEAKER")
        elif event_type == "response.audio_transcript.delta":
            # Accumulate transcript and update dialogue box with the full sentence
            if "delta" in event:
                self._current_transcript += event["delta"]
                self.npc_message = self._current_transcript
                debug_print(f"[VOICE_TEXT_INSTANT] 🎙️⚡ VOICE-TO-TEXT: '{event['delta']}'", "VOICE_TEXT_INSTANT")
        elif event_type == "response.audio_transcript.done":
            # Optionally finalize the transcript
            pass
        elif event_type == "conversation.item.create":
            debug_print(f"[SERVER_EVENT] 📨 conversation.item.create: {event}", "SERVER_EVENT")
        elif event_type == "input_text":
            debug_print(f"[SERVER_EVENT] 📝 input_text: {event}", "SERVER_EVENT")
        elif event_type == "input_audio_buffer.speech_started":
            debug_print("🎤 User speech detected by server", "USER_SPEECH")
        elif event_type == "input_audio_buffer.speech_stopped":
            debug_print("🔇 User speech ended", "USER_SPEECH")
        elif event_type == "response.created":
            response_id = event.get('response', {}).get('id', 'unknown')
            debug_print(f"[EVENT] response.created: {response_id}", "RESPONSE_EVENT")
            # Only cancel if there's actually an active response with different ID
            if self.response_in_progress and self.current_response_id and self.current_response_id != response_id:
                debug_print(f"⚠️ CANCELLING PREVIOUS RESPONSE - New: {response_id}, Previous: {self.current_response_id}", "RESPONSE_CONFLICT")
                await self.cancel_ai_response()  # Cancel the previous response
                self.audio_handler.stop_playback()
                self.is_speaking = False
                self.audio_buffer = b''  # Clear any pending audio
            # Set new response as current
            self.current_response_id = response_id
            self.response_in_progress = True
            self._current_transcript = ""  # Reset transcript buffer for new response
            debug_print(f"🆕 New AI response starting: {response_id}", "RESPONSE_START")
        elif event_type == "response.done":
            response_id = event.get('response', {}).get('id', 'unknown')
            debug_print(f"[EVENT] response.done: {response_id}", "RESPONSE_EVENT")
            if response_id == self.current_response_id:
                self.response_in_progress = False
                self.current_response_id = None
                debug_print(f"✅ Response completed: {response_id}", "RESPONSE_COMPLETE")
            else:
                debug_print(f"⚠️ Mismatched response completion: {response_id} vs {self.current_response_id}", "RESPONSE_MISMATCH")
        else:
            # Log unknown event types to see what we might be missing
            debug_print(f"❓ Unknown event: {event_type} - {str(event)[:200]}...", "WS_UNKNOWN")

    async def send_audio_chunk(self, audio_data):
        """Send audio chunk to the realtime API."""
        if self.ws and audio_data:
            base64_chunk = base64.b64encode(audio_data).decode('utf-8')
            await self.send_event({
                "type": "input_audio_buffer.append",
                "audio": base64_chunk
            })

    async def commit_audio_buffer(self, trigger_source="VAD/voice"):  # Add trigger_source for debug
        """Commit the audio buffer and trigger AI response."""
        if self.ws:
            if self.response_in_progress or self.is_speaking or self.ai_response_lock.locked():
                debug_print(f"⏳ Skipping commit_audio_buffer from {trigger_source}: response already in progress or AI speaking", "AUDIO_COMMIT")
                return
            with self.ai_response_lock:
                self.response_in_progress = True
                try:
                    # Transcribe the audio buffer before sending to AI
                    debug_print("📝 Transcribing user audio buffer before sending to AI...", "ASR")
                    transcript = self.transcribe_audio_buffer(self.audio_buffer)
                    debug_print(f"🗣️ User transcript: '{transcript}'", "ASR")
                    if transcript.strip():
                        await self.send_text_message(transcript, trigger_source="voice_transcript")
                    else:
                        debug_print("⚠️ No transcript detected, not sending empty message to AI.", "ASR")
                    self.audio_buffer = b''
                    # Force garbage collection after processing audio buffer
                    import gc
                    gc.collect()
                except Exception as e:
                    debug_print(f"❌ Error in commit_audio_buffer: {e}", "AUDIO_COMMIT")
                    self.response_in_progress = False
                    # Clear audio buffer on error to prevent corruption
                    self.audio_buffer = b''

    def transcribe_audio_buffer(self, audio_buffer):
        """Transcribe PCM16 audio buffer to text using OpenAI Whisper (or fallback)."""
        import openai
        import tempfile
        import os
        import wave
        transcript = ""
        tmpfile_path = None
        try:
            # Check if buffer is too small or empty
            if len(audio_buffer) < 1024:  # Less than 1KB
                debug_print("⚠️ Audio buffer too small for transcription", "ASR_ERROR")
                return ""
            
            # Save buffer to temp WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                wf = wave.open(tmpfile, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_buffer)
                wf.close()
                tmpfile_path = tmpfile.name
            
            # Use OpenAI Whisper ASR
            with open(tmpfile_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                ).text
            
        except Exception as e:
            debug_print(f"❌ ASR error: {e}", "ASR_ERROR")
        finally:
            # CRITICAL: Always clean up temp file
            if tmpfile_path and os.path.exists(tmpfile_path):
                try:
                    os.remove(tmpfile_path)
                    debug_print(f"🗑️ Cleaned up temp file: {tmpfile_path}", "ASR_CLEANUP")
                except Exception as cleanup_error:
                    debug_print(f"⚠️ Error cleaning up temp file: {cleanup_error}", "ASR_CLEANUP")
        
        return transcript

    async def cancel_ai_response(self):
        """Cancel the current AI response."""
        if self.ws:
            await self.send_event({"type": "response.cancel"})

    def start_continuous_recording(self):
        """Start continuous recording for seamless voice interaction."""
        debug_print("🎙️ Starting continuous voice recording for seamless interaction...", "CONTINUOUS_VOICE")
        if not self.is_listening:
            self.is_listening = True
            self.audio_handler.start_recording()
            
            # Start continuous audio streaming
            self.audio_thread = threading.Thread(target=self._stream_audio_with_vad)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            debug_print("✅ Continuous voice recording active!", "CONTINUOUS_VOICE")

    async def _continuous_voice_monitoring(self):
        """Continuously monitor for voice activity - auto-interruption disabled to prevent issues."""
        debug_print("👂 Starting continuous voice activity monitoring (auto-interruption disabled)...", "VAD")
        while self.active and self.continuous_recording:
            try:
                # Skip voice monitoring if in text mode
                if self.text_mode_active:
                    await asyncio.sleep(0.2)  # Sleep longer when in text mode
                    continue
                    
                await asyncio.sleep(0.05)  # REDUCED from 0.1 to 0.05 for more responsive voice monitoring
            except Exception as e:
                debug_print(f"❌ Error in voice monitoring: {e}", "VAD_ERROR")
                await asyncio.sleep(0.05)  # REDUCED from 0.1 to 0.05 for more responsive voice monitoring

    async def _user_silence_nudge(self):
        silence_nudge_delay = 3.0  # seconds
        nudge_sent = False
        last_ai_spoke = time.time()
        last_user_spoke = time.time()
        while self.active:
            await asyncio.sleep(0.1)  # REDUCED from 0.2 to 0.1 for more responsive nudge detection
            if self.is_user_speaking:
                last_user_spoke = time.time()
                nudge_sent = False
                self.waiting_for_user_input = False  # User spoke, reset flag
            if self.is_speaking or self.response_in_progress or self.ai_response_lock.locked():
                last_ai_spoke = time.time()
                nudge_sent = False
            if not self.is_speaking and not self.is_user_speaking and not self.response_in_progress and not self.ai_response_lock.locked():
                if (time.time() - last_ai_spoke > 0.5) and (time.time() - last_user_spoke > silence_nudge_delay):
                    if not nudge_sent and not self.waiting_for_user_input:
                        nudge_text = "Let me know if you have any questions, or if you'd like to continue our conversation!"
                        debug_print(f"🤖 Sending gentle nudge: {nudge_text}", "NUDGE")
                        await self.send_text_message(nudge_text, trigger_source="nudge")
                        nudge_sent = True
                        self.waiting_for_user_input = True  # Block further nudges until user speaks

    def _stream_audio_with_vad(self):
        """Stream audio chunks with interruption detection and cooldown logic."""
        debug_print("🔊 Starting audio streaming with interruption detection...", "AUDIO_STREAM")
        
        while self.is_listening and self.audio_handler.is_recording and self.active:
            try:
                # Skip voice processing if in text mode
                if self.text_mode_active:
                    time.sleep(0.1)  # Sleep longer when in text mode to save CPU
                    continue
                    
                chunk = self.audio_handler.record_chunk()
                if chunk and self.loop:
                    # Perform voice activity detection for interruption and UI feedback
                    try:
                        audio_level = self._calculate_audio_level(chunk)
                        self._process_voice_activity(audio_level)
                    except Exception as vad_error:
                        debug_print(f"⚠️ Error in voice activity detection: {vad_error}", "VAD_ERROR")
                        # Continue with default audio level
                        audio_level = 0.0
                        self.last_audio_level = audio_level
                    
                    # Always send audio chunks to server for processing (no cooldown check)
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.send_audio_chunk(chunk), 
                            self.loop
                        )
                    except Exception as send_error:
                        debug_print(f"⚠️ Error sending audio chunk: {send_error}", "AUDIO_STREAM_ERROR")
                
                time.sleep(0.002)  # REDUCED to 2ms delay for maximum real-time performance
            except Exception as e:
                debug_print(f"❌ Error in audio streaming: {e}", "AUDIO_ERROR")
                # Longer delay on error to prevent rapid error loops
                time.sleep(0.1)
                
                # If we get too many errors, stop streaming
                if not hasattr(self, '_audio_error_count'):
                    self._audio_error_count = 0
                self._audio_error_count += 1
                
                if self._audio_error_count > 10:
                    debug_print("🚨 Too many audio streaming errors - stopping", "AUDIO_ERROR")
                    self.is_listening = False
                    # Force garbage collection to clean up any corrupted audio data
                    import gc
                    gc.collect()
                    break
        
        debug_print("🔇 Audio streaming stopped", "AUDIO_STREAM")

    def _calculate_audio_level(self, audio_chunk):
        """Calculate the audio level for voice activity detection with voice filtering."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            if len(audio_array) == 0:
                return 0.0
                
            sample_rate = 24000  # Our recording sample rate
            fft = np.fft.rfft(audio_array)
            freqs = np.fft.rfftfreq(len(audio_array), 1/sample_rate)
            
            # Expanded voice frequency range for better detection
            voice_mask = (freqs >= 80) & (freqs <= 4000)  # Broader range for all voice types
            voice_energy = np.sum(np.abs(fft[voice_mask]) ** 2)
            total_energy = np.sum(np.abs(fft) ** 2)
            
            # Calculate RMS for overall audio level
            rms = np.sqrt(np.mean(audio_array ** 2))
            normalized_level = min(rms / 32767.0, 1.0)
            
            # More lenient voice detection for interruption
            if total_energy > 0:
                voice_ratio = voice_energy / total_energy
                
                # IMPROVED: More selective voice detection for interruption
                if voice_ratio > 0.25 and normalized_level > self.noise_floor:  # INCREASED from 0.2 for higher confidence
                    # Boost the level for voice-like sounds, but less aggressively
                    voice_boost = 1.0 + (voice_ratio * 0.3)  # REDUCED from 0.5 for more conservative detection
                    final_level = normalized_level * voice_boost
                    
                    # Debug output for tuning (only for significant levels)
                    if final_level > 0.08:  # INCREASED threshold for debug output
                        debug_print(f"🎤 Voice detected: level={final_level:.3f}, ratio={voice_ratio:.2f}, raw={normalized_level:.3f}", "VAD_VOICE")
                    
                    return min(final_level, 1.0)
                else:
                    # Filter out non-voice sounds
                    if normalized_level > 0.05:
                        debug_print(f"🔇 Filtered non-voice: level={normalized_level:.3f}, ratio={voice_ratio:.2f}", "VAD_FILTER")
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            debug_print(f"⚠️ Audio level calculation error: {e}", "VAD_ERROR")
            # Fallback to simple RMS calculation
            try:
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array ** 2))
                return min(rms / 32767.0, 1.0)
            except:
                return 0.0

    def _process_voice_activity(self, audio_level):
        self.last_audio_level = audio_level
        current_threshold = self.voice_activity_threshold
        vad_cooldown = 0.3  # seconds after AI speech to ignore VAD
        now = time.time()
        
        # If AI is speaking, make VAD easier to trigger for interruption (FIXED)
        if self.is_speaking or self.audio_playing:
            current_threshold = self.voice_activity_threshold * 0.85  # INCREASED from 0.7 - requires more power to interrupt
            self.ai_speaking_pause = True
            debug_print(f"🔊 AI SPEAKING - VAD threshold for interruption: {current_threshold:.2f}", "VAD_AI_SPEAKING")
        # If AI just finished speaking, ignore VAD for a short cooldown
        elif now - getattr(self, 'last_ai_speaking_end', 0) < vad_cooldown:
            debug_print(f"⏳ VAD cooldown after AI speech ({now - self.last_ai_speaking_end:.2f}s)", "VAD_COOLDOWN")
            return  # Ignore VAD triggers during cooldown
        else:
            self.ai_speaking_pause = False
            
        # Check if audio level exceeds threshold
        if audio_level > current_threshold:
            if not self.is_user_speaking:
                self.is_user_speaking = True
                self.silence_duration = 0
                self.speech_start_time = time.time()
                self.waiting_for_user_input = False  # User started speaking, allow nudges again
                
                # IMMEDIATE interruption when AI is speaking
                if self.is_speaking or self.audio_playing:
                    debug_print("🚨 User speech detected: INSTANTLY interrupting AI!", "VAD_INTERRUPT")
                    self.interrupt_ai()  # Interrupt AI instantly on speech start
                else:
                    debug_print("🎤 User speech started", "VAD_START")
                    
            self.silence_duration = 0
        else:
            self.speech_confirmation_buffer = []
            if self.is_user_speaking:
                self.silence_duration += 0.01
                if self.silence_duration >= self.max_silence_duration:
                    speech_time = 0
                    if self.speech_start_time:
                        speech_time = time.time() - self.speech_start_time
                    debug_print(f"⏸️ USER SPEECH ENDED (silence: {self.silence_duration:.2f}s, speech_time: {speech_time:.2f}s)", "VAD_END")
                    if speech_time >= self.min_speech_duration:
                        self.is_user_speaking = False
                        self.silence_duration = 0
                        if self.loop and self.active:
                            debug_print("📤 Committing audio buffer after real user speech ended.", "AUDIO_COMMIT")
                            import asyncio
                            asyncio.run_coroutine_threadsafe(self.commit_audio_buffer(trigger_source="VAD/voice_end"), self.loop)
                    else:
                        debug_print(f"🛑 Speech too short ({speech_time:.2f}s), not committing.", "VAD_END")
                        self.is_user_speaking = False
                        self.silence_duration = 0
                        self.speech_start_time = None

    async def send_text_message(self, text_content, trigger_source="user_input/nudge"):
        """Send a text message and get voice response via realtime API."""
        if self.ws:
            if self.response_in_progress or self.is_speaking or self.ai_response_lock.locked():
                debug_print(f"⏳ Skipping send_text_message from {trigger_source}: response already in progress or AI speaking", "TEXT_SEND")
                return
            with self.ai_response_lock:
                self.response_in_progress = True
                try:
                    debug_print(f"📝 Sending text message: {text_content[:50]}... (trigger: {trigger_source})", "TEXT_SEND")
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
                    # Trigger AI response with optimized settings for INSTANT text display
                    debug_print("⚡ Requesting INSTANT response with text priority...", "TEXT_SEND")
                    await self.send_event({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            # Optimize for instant text display
                            "instructions": "Respond immediately with text transcript. Prioritize speed.",
                            "temperature": 0.6  # Lower for faster, more focused responses
                        }
                    })
                    debug_print(f"✅ Text message sent with INSTANT priority: {text_content[:30]}...", "TEXT_SEND")
                except Exception as e:
                    debug_print(f"❌ Error in send_text_message: {e}", "TEXT_SEND")
                    self.response_in_progress = False

    def send_text_message_sync(self, text_content, trigger_source="user_input_sync"):
        """Synchronous wrapper to send text message."""
        if not text_content.strip():
            return
        if self.response_in_progress or self.is_speaking or self.ai_response_lock.locked():
            debug_print(f"⏳ Skipping send_text_message_sync from {trigger_source}: response already in progress or AI speaking", "TEXT_SEND")
            return
        user_message = text_content.strip()
        debug_print(f"👤 User says: \"{user_message}\" (trigger: {trigger_source})", "SPEAKER")
        # If AI is speaking, interrupt it first
        if self.is_speaking:
            debug_print("✋ Interrupting AI for text input...", "VOICE_REC")
            self.interrupt_ai()
        # Store the message for history
        self.conversation_history.append({"role": "user", "content": user_message})
        # Send via WebSocket if available, otherwise use fallback
        if self.loop and self.ws:
            asyncio.run_coroutine_threadsafe(
                self.send_text_message(user_message, trigger_source=trigger_source), 
                self.loop
            )
        else:
            # Fallback to direct OpenAI API if WebSocket not available
            self.send_text_fallback(user_message)
        # Clear input
        self.user_input = ""

    def send_text_fallback(self, text_content):
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
            debug_print(f"🎵 TTS called with text: {text[:50]}...", "TTS_START")
            self.npc_message = text  # Always set before TTS so dialogue box matches speech
            debug_print(f"🎭 Using voice: {self.get_tts_voice_for_npc()}", "TTS_START")
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.get_tts_voice_for_npc(),
                input=text
            )
            debug_print(f"✅ TTS API call successful, got {len(response.content)} bytes", "TTS_SUCCESS")
            
            audio_data = response.content  # MP3 bytes
            # Decode MP3 to PCM16
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            pcm_data = audio_segment.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data
            debug_print(f"🔄 Audio converted to PCM16: {len(pcm_data)} bytes", "TTS_SUCCESS")
            
            self.is_speaking = True
            debug_print(f"🎧 Calling audio_handler.play_audio with dialogue_system={self}", "TTS_SUCCESS")
            self.audio_handler.play_audio(pcm_data, dialogue_system=self)
            debug_print(f"✅ TTS completed successfully", "TTS_SUCCESS")
        except Exception as e:
            debug_print(f"❌ TTS error: {e}", "TTS_ERROR")

    def get_tts_voice_for_npc(self):
        """Get TTS voice for current NPC using AI_AGENTS data."""
        if self.current_npc in AI_AGENTS:
            return AI_AGENTS[self.current_npc]["voice"]
        
        # Fallback voices if not in AI_AGENTS
        if self.current_npc == "HR":
            return "nova"  # Female voice for HR Director Sarah
        else:  # CEO
            return "onyx"   # Male voice for CEO Michael

    def stop_continuous_recording(self):
        """Stop continuous recording when ending conversation."""
        if self.is_listening:
            debug_print("⏹️ Stopping continuous voice recording...", "CONTINUOUS_VOICE")
            self.is_listening = False
            self.continuous_recording = False
            self.is_user_speaking = False
            self.audio_handler.stop_recording()
            debug_print("✅ Continuous voice recording stopped", "CONTINUOUS_VOICE")

    def end_conversation(self):
        """Properly end the seamless conversation and clean up all resources."""
        try:
            # --- Set inactive immediately so all threads exit ---
            self.active = False
            
            # Show conversation end with participants
            try:
                if self.current_npc:
                    speaker_name = "Sarah (HR Director)" if self.current_npc == "HR" else "Michael (CEO)"
                    debug_print("=" * 60, "CONVERSATION")
                    debug_print(f"🏁 ENDING SEAMLESS CONVERSATION WITH {speaker_name.upper()}", "CONVERSATION")
                    debug_print(f"👋 User has left the seamless conversation with {speaker_name}", "CONVERSATION")
                    debug_print("=" * 60, "CONVERSATION")
                debug_print("🔚 Ending seamless conversation...", "DIALOGUE")
            except Exception as e:
                debug_print(f"[CLEANUP] Error displaying conversation end: {e}", "CLEANUP_ERROR")
            
            # Stop continuous recording
            try:
                self.stop_continuous_recording()
            except Exception as e:
                debug_print(f"[CLEANUP] Error stopping continuous recording: {e}", "CLEANUP_ERROR")
            
            # Stop any AI speech
            try:
                if self.is_speaking or (hasattr(self.audio_handler, 'is_playing') and self.audio_handler.is_playing):
                    debug_print("[DIALOGUE] end_conversation: Stopping AI speech playback.", "AUDIO_DEBUG")
                    self.is_speaking = False
                    self.audio_handler.stop_playback()
                else:
                    debug_print("[DIALOGUE] end_conversation: No AI speech to stop.", "AUDIO_DEBUG")
            except Exception as e:
                debug_print(f"[CLEANUP] Error stopping AI speech: {e}", "CLEANUP_ERROR")
            
            # Clear buffers and text input
            try:
                self.audio_buffer = b''
                self.npc_message = ""
                self.user_input = ""
                self.conversation_history = []
                # Clear audio queue to prevent memory leaks
                if hasattr(self, 'audio_queue'):
                    queue_size = len(self.audio_queue)
                    self.audio_queue.clear()
                    debug_print(f"🗑️ Cleared {queue_size} queued audio responses", "CLEANUP")
                # Reset voice activity detection state
                self.is_user_speaking = False
                self.silence_duration = 0
                self.last_audio_level = 0.0
                # Reset mode to voice mode for next conversation
                self.text_mode_active = False
                # CRITICAL: Force garbage collection to prevent memory leaks
                import gc
                gc.collect()
                debug_print("🧹 Forced garbage collection during cleanup", "CLEANUP")
            except Exception as e:
                debug_print(f"[CLEANUP] Error clearing buffers: {e}", "CLEANUP_ERROR")
            
            # *** CRITICAL FIX *** Close WebSocket connection
            try:
                if self.ws:
                    debug_print("🔌 Closing WebSocket connection...", "CLEANUP")
                    if self.loop:
                        import asyncio
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._close_websocket(), 
                                self.loop
                            ).result(timeout=2.0)  # Add timeout
                        except Exception as e:
                            debug_print(f"[CLEANUP] Error in WebSocket close coroutine: {e}", "CLEANUP_ERROR")
                    self.ws = None
            except Exception as e:
                debug_print(f"[CLEANUP] Error closing WebSocket: {e}", "CLEANUP_ERROR")
            
            # Reset conversation-specific state
            try:
                self.current_npc = None
                self.loop = None
                self.response_in_progress = False
                self.current_response_id = None
                self.interrupting = False
                self.audio_playing = False
                self.ai_speaking_pause = False
            except Exception as e:
                debug_print(f"[CLEANUP] Error resetting state: {e}", "CLEANUP_ERROR")
            
            # Clean up audio handler (initialize new one for next conversation)
            try:
                self.audio_handler.cleanup()
                # Small delay before creating new handler
                import time
                time.sleep(0.1)
                # Force garbage collection before creating new handler
                import gc
                gc.collect()
                debug_print("🧹 Forced garbage collection before creating new audio handler", "CLEANUP")
                self.audio_handler = AudioHandler()  # Fresh audio handler for next conversation
            except Exception as e:
                debug_print(f"[CLEANUP] Error cleaning up audio handler: {e}", "CLEANUP_ERROR")
                # Try to create a new audio handler anyway
                try:
                    import gc
                    gc.collect()  # Force cleanup before retry
                    self.audio_handler = AudioHandler()
                except Exception as e2:
                    debug_print(f"[CLEANUP] Error creating new audio handler: {e2}", "CLEANUP_ERROR")
            
            debug_print("✅ Seamless conversation ended and cleaned up", "CLEANUP")
            
        except Exception as e:
            debug_print(f"[CLEANUP] FATAL error in end_conversation: {e}", "CLEANUP_FATAL")
            # Emergency cleanup
            try:
                self.active = False
                self.is_speaking = False
                self.audio_playing = False
                self.interrupting = False
                if hasattr(self, 'audio_handler'):
                    self.audio_handler.is_playing = False
                    self.audio_handler.is_recording = False
            except:
                pass

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
        try:
            if self.is_speaking or self.audio_playing:
                debug_print("🛑 INTERRUPTING AI SPEECH - IMMEDIATE STOP", "INTERRUPT")
                
                # Set flags immediately to stop all AI activity
                self.is_speaking = False
                self.audio_playing = False
                self.interrupting = True
                self.ai_speaking_pause = False
                
                # Show interruption in UI immediately
                self.npc_message = "[Interrupted by user]"
                
                # IMMEDIATELY stop audio playback (synchronous) with error handling
                try:
                    if hasattr(self.audio_handler, 'is_playing') and self.audio_handler.is_playing:
                        debug_print("🔇 Stopping audio playback immediately...", "INTERRUPT")
                        self.audio_handler.is_playing = False
                        self.audio_handler._stop_event.set()
                        
                        # Force close the audio stream immediately
                        if hasattr(self.audio_handler, 'playback_stream') and self.audio_handler.playback_stream:
                            try:
                                self.audio_handler.playback_stream.stop_stream()
                                self.audio_handler.playback_stream.close()
                                debug_print("✅ Audio stream force-closed", "INTERRUPT")
                            except Exception as e:
                                debug_print(f"⚠️ Error force-closing audio stream: {e}", "INTERRUPT")
                            finally:
                                self.audio_handler.playback_stream = None
                except Exception as e:
                    debug_print(f"⚠️ Error stopping audio playback: {e}", "INTERRUPT")
                
                # Cancel AI response via WebSocket with error handling
                try:
                    if self.loop and self.ws:
                        debug_print("📡 Cancelling AI response via WebSocket...", "INTERRUPT")
                        import asyncio
                        fut = asyncio.run_coroutine_threadsafe(self.cancel_ai_response(), self.loop)
                        fut.result(timeout=0.5)  # Shorter timeout for faster interruption
                        debug_print("✅ AI response cancelled", "INTERRUPT")
                except Exception as e:
                    debug_print(f"⚠️ Error cancelling AI response: {e}", "INTERRUPT")
                
                # Clear all audio buffers and queues with error handling
                try:
                    self.audio_buffer = b''
                    if hasattr(self, 'audio_queue'):
                        self.audio_queue.clear()
                        debug_print(f"🗑️ Cleared {len(self.audio_queue)} queued audio responses", "INTERRUPT")
                    if hasattr(self, '_current_transcript'):
                        self._current_transcript = ""
                    # CRITICAL: Force garbage collection to prevent memory leaks
                    import gc
                    gc.collect()
                    debug_print("🧹 Forced garbage collection after interruption", "INTERRUPT")
                except Exception as e:
                    debug_print(f"⚠️ Error clearing buffers: {e}", "INTERRUPT")
                
                # Reset response tracking
                self.response_in_progress = False
                self.current_response_id = None
                
                # Clear interruption flag after a short delay with error handling
                import threading
                def clear_interruption():
                    try:
                        import time
                        time.sleep(0.5)
                        self.interrupting = False
                        self.npc_message = ""
                        debug_print("✅ Interruption completed - Ready for user input", "INTERRUPT")
                    except Exception as e:
                        debug_print(f"⚠️ Error in clear_interruption: {e}", "INTERRUPT")
                
                try:
                    threading.Thread(target=clear_interruption, daemon=True).start()
                except Exception as e:
                    debug_print(f"⚠️ Error starting clear_interruption thread: {e}", "INTERRUPT")
                    # Fallback: clear immediately
                    self.interrupting = False
                    self.npc_message = ""
                
            else:
                debug_print("ℹ️ No AI speech to interrupt", "INTERRUPT")
        except Exception as e:
            debug_print(f"❌ CRITICAL ERROR in interrupt_ai: {e}", "INTERRUPT_ERROR")
            # Emergency cleanup
            try:
                self.is_speaking = False
                self.audio_playing = False
                self.interrupting = False
                self.ai_speaking_pause = False
                self.npc_message = ""
            except:
                pass

    def on_user_speech_start(self):
        # Called as soon as user speech is detected
        debug_print("🛑 Interrupting AI: User speech started! Now listening to user.", "INTERRUPT")
        self.interrupt_ai()  # Immediately stop AI speech

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

            # Get NPC name from AI_AGENTS definitions
            if self.current_npc in AI_AGENTS:
                agent = AI_AGENTS[self.current_npc]
                npc_name = f"{agent['name']} ({agent['title']})"
            else:
                npc_name = "Sarah (HR)" if self.current_npc == "HR" else "Michael (CEO)"
            
            # Show status based on current mode (voice vs text)
            if self.text_mode_active:
                # Text mode instructions
                if self.interrupting:
                    instruction_text = f"⏳ INTERRUPTING AI... Please wait."
                elif self.is_speaking:
                    voice_indicator = "👩‍💼" if self.current_npc == "HR" else "👨‍💼" 
                    instruction_text = f"{voice_indicator} {npc_name} is speaking | ⌨️ TEXT MODE | TAB to switch to voice | Shift+Q: exit"
                else:
                    instruction_text = f"⌨️ TEXT MODE with {npc_name} | Type + ENTER to send | TAB to switch to voice | Shift+Q: exit"
                
                # Show text mode controls
                controls_text = f"📝 TEXT CONTROLS: Type + ENTER to send | BACKSPACE to delete | SPACE for space | TAB to switch to voice"
                controls_surface = self.font.render(controls_text, True, (100, 255, 255))  # Cyan for text mode
                self.ui_surface.blit(controls_surface, (40, box_y + 35))
                
            else:
                # Voice mode instructions (original)
                if self.interrupting:
                    instruction_text = f"⏳ INTERRUPTING AI... Please wait."
                elif self.is_user_speaking:
                    instruction_text = f"🗣️ YOU ARE SPEAKING to {npc_name} | 🎙️ VOICE MODE | TAB to switch to text | Shift+Q: exit"
                elif self.is_speaking:
                    voice_indicator = "👩‍💼" if self.current_npc == "HR" else "👨‍💼" 
                    protection_status = "🔒 PROTECTED" if self.ai_speaking_pause else "🔓 INTERRUPTIBLE"
                    instruction_text = f"{voice_indicator} {npc_name} is speaking {protection_status} | 🎙️ Speak OR TAB to switch | Shift+Q: exit"
                else:
                    # Show current audio level for feedback with threshold info
                    audio_level_bar = "▓" * int(self.last_audio_level * 20) + "░" * (20 - int(self.last_audio_level * 20))
                    threshold_met = "✅" if self.last_audio_level > self.voice_activity_threshold else "❌"
                    instruction_text = f"🎙️ VOICE MODE with {npc_name} | Speak OR TAB to switch | Audio: [{audio_level_bar}] {threshold_met} | Shift+Q: exit"
                    
                    # Add voice mode debug info
                    debug_text = f"🎤 Voice Level: {self.last_audio_level:.3f} | Threshold: {self.voice_activity_threshold:.3f} | TAB to switch to text mode"
                    debug_surface = self.font.render(debug_text, True, (255, 255, 100))  # Yellow for voice mode
                    self.ui_surface.blit(debug_surface, (40, box_y + 35))
                
            instruction_surface = self.font.render(instruction_text, True, (255, 255, 255))
            self.ui_surface.blit(instruction_surface, (40, box_y + 10))

            # NPC message in white with header - OPTIMIZED FOR INSTANT DISPLAY
            if self.npc_message:
                # Add a label to make it clear this is the NPC's response
                if self.current_npc in AI_AGENTS:
                    agent = AI_AGENTS[self.current_npc]
                    npc_label = f"{agent['name']}:"
                else:
                    npc_label = f"{npc_name}:"
                label_surface = self.font.render(npc_label, True, (200, 200, 255))  # Light blue for label
                self.ui_surface.blit(label_surface, (40, box_y + 40))
                
                # Render the actual response text below the label - INSTANT DISPLAY
                self.render_text(self.ui_surface, self.npc_message, 40, box_y + 65)
                
                # Show real-time text length for debugging lag
                text_length = len(self.npc_message)
                if text_length > 0:
                    length_text = f"[{text_length} chars] {'🎙️VOICE' if self.is_speaking else '📝TEXT'}"
                    length_surface = self.font.render(length_text, True, (100, 255, 100))  # Green status
                    self.ui_surface.blit(length_surface, (WINDOW_WIDTH - 200, box_y + 10))
            else:
                # Show real-time status for debugging
                if self.is_speaking:
                    debug_text = f"🔊 AI SPEAKING - Waiting for text... (Voice active: {self.is_speaking})"
                    debug_surface = self.font.render(debug_text, True, (255, 255, 0))  # Yellow debug text
                    self.ui_surface.blit(debug_surface, (40, box_y + 40))
                else:
                    # Show current status with timestamp
                    import time
                    current_time = time.strftime("%H:%M:%S")
                    status_text = f"[{current_time}] Ready for {npc_name} response..."
                    status_surface = self.font.render(status_text, True, (128, 128, 128))  # Gray status text
                    self.ui_surface.blit(status_surface, (40, box_y + 40))

            # Text input field with mode indicator
            if self.input_active:
                # Show mode indicator in the input field
                if self.text_mode_active:
                    mode_indicator = "📝 TEXT MODE"
                    input_prompt = f"{mode_indicator} > {self.user_input}_"
                    input_color = (100, 255, 255)  # Cyan for text mode
                else:
                    mode_indicator = "🎙️ VOICE MODE"
                    input_prompt = f"{mode_indicator} > {self.user_input}_"
                    input_color = (255, 255, 100)  # Yellow for voice mode
                
                input_surface = self.font.render(input_prompt, True, input_color)
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
        if not self.active:
            return

        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            
            # Check for Shift+Q to exit chat
            if keys[pygame.K_LSHIFT] and event.key == pygame.K_q:
                self.interrupt_ai()  # Stop all AI speech and clear buffers instantly
                self.end_conversation()  # Properly clean up resources
                debug_print("👋 User ended seamless conversation", "CONVERSATION_END")
                # Return both the command and the initial position
                return {"command": "move_player_back", "initial_pos": self.initial_player_pos}

            # Handle text input when input is active (backup method)
            if self.input_active:
                if event.key == pygame.K_RETURN and self.user_input.strip():
                    # Send text message
                    self.send_text_message_sync(self.user_input)
                elif event.key == pygame.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                elif event.key == pygame.K_SPACE:
                    # SPACE key for instant interruption or adding space in text mode
                    if self.is_speaking or self.audio_playing:
                        # If AI is speaking, interrupt first
                        debug_print("⚡ SPACE key pressed - Instant interruption!", "KEYBOARD_INTERRUPT")
                        self.interrupt_ai()
                    elif self.text_mode_active:
                        # In text mode, SPACE adds a space character
                        self.user_input += " "
                elif event.key == pygame.K_TAB:
                    # TAB key toggles between voice and text modes
                    # Toggle between voice and text modes
                    if self.text_mode_active:
                        # Switch to voice mode
                        debug_print("🎙️ TAB pressed - Switching to VOICE MODE", "MODE_SWITCH")
                        self.text_mode_active = False
                        self.user_input = ""  # Clear text input
                        
                        # Restart continuous recording
                        if not self.is_listening:
                            self.start_continuous_recording()
                        
                        debug_print("✅ Voice mode activated", "MODE_SWITCH")
                    else:
                        # Switch to text mode
                        debug_print("⌨️ TAB pressed - Switching to TEXT MODE", "MODE_SWITCH")
                        self.text_mode_active = True
                        
                        # Stop continuous voice recording
                        self.stop_continuous_recording()
                        
                        # Clear any voice activity state
                        self.is_user_speaking = False
                        self.last_audio_level = 0.0
                        
                        debug_print("✅ Text mode activated", "MODE_SWITCH")
                elif event.unicode.isprintable() and not keys[pygame.K_LCTRL]:
                    # Only allow text input when in text mode
                    if self.text_mode_active:
                        # Add character to input (but not if Ctrl is held)
                        self.user_input += event.unicode
                        debug_print(f"⌨️ Text input: '{self.user_input}'", "TEXT_MODE")
                    else:
                        # In voice mode, typing switches to text mode
                        debug_print("⌨️ TEXT MODE ACTIVATED - Stopping voice interaction", "MODE_SWITCH")
                        self.text_mode_active = True
                        
                        # Stop continuous voice recording
                        self.stop_continuous_recording()
                        
                        # Interrupt any AI speech
                        if self.is_speaking or self.audio_playing:
                            debug_print("⚡ Interrupting AI for text mode switch!", "MODE_SWITCH")
                            self.interrupt_ai()
                        
                        # Clear any voice activity state
                        self.is_user_speaking = False
                        self.last_audio_level = 0.0
                        
                        # Add the typed character
                        self.user_input += event.unicode
                        debug_print(f"⌨️ Text input: '{self.user_input}'", "TEXT_MODE")

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
            debug_print("❌ CRITICAL: No API key found in .env file", "WEBSOCKET_ERROR")
            return
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
            "Content-Type": "application/json"
        }
        
        try:
            debug_print(f"🔌 Attempting WebSocket connection to {self.url}", "WEBSOCKET")
            debug_print(f"🤖 Using model: {self.model}", "WEBSOCKET")
            debug_print(f"🔑 API key: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}", "WEBSOCKET")
            
            # Try different header parameter names for different websockets versions
            debug_print("🔧 Trying WebSocket connection methods...", "WEBSOCKET")
            try:
                debug_print("   📡 Attempting with additional_headers (websockets >= 12.0)...", "WEBSOCKET")
                self.ws = await websockets.connect(
                    f"{self.url}?model={self.model}",
                    additional_headers=headers,
                    ssl=self.ssl_context,
                    ping_interval=20,  # Keep connection alive
                    ping_timeout=20
                )
                debug_print("   ✅ Connected with additional_headers method", "WEBSOCKET")
            except TypeError:
                debug_print("   ⚠️ Fallback to extra_headers (older websockets)...", "WEBSOCKET")
                self.ws = await websockets.connect(
                    f"{self.url}?model={self.model}",
                    extra_headers=headers,
                    ssl=self.ssl_context,
                    ping_interval=20,
                    ping_timeout=20
                )
                debug_print("   ✅ Connected with extra_headers method", "WEBSOCKET")
            
            debug_print("🎉 SUCCESS! Connected to OpenAI Realtime API", "WEBSOCKET_SUCCESS")
            
            # Configure session
            npc_voice = self.get_voice_for_npc()
            debug_print(f"🎭 Using voice '{npc_voice}' for {self.current_npc}", "WEBSOCKET")
            
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": self.get_instructions_for_npc() + "\n\nIMPORTANT: Respond immediately and naturally. Keep responses conversational and under 3 sentences for real-time flow. Generate text transcripts instantly.",
                    "voice": npc_voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.05
                    },
                    "temperature": 0.6,
                    "max_response_output_tokens": 300
                }
            }
            debug_print("📤 Sending session configuration...", "WEBSOCKET")
            debug_print(f"   📋 Config: {json.dumps(session_config, indent=2)}", "WEBSOCKET")
            await self.send_event(session_config)
            
            # --- REMOVED: Do NOT send initial greeting message or request AI to speak it ---
            # This ensures only your custom greeting is spoken and shown.
            # ---
            debug_print("✅ Conversation initiated with custom greeting only!", "WEBSOCKET_SUCCESS")
            
        except Exception as e:
            debug_print(f"❌ WebSocket connection FAILED: {str(e)}", "WEBSOCKET_ERROR")
            debug_print(f"🔍 Error type: {type(e).__name__}", "WEBSOCKET_ERROR")
            import traceback
            debug_print(f"📋 Full traceback: {traceback.format_exc()}", "WEBSOCKET_ERROR")
            self.ws = None

    async def send_event(self, event):
        """Send an event to the WebSocket server."""
        try:
            if self.ws:
                # Add timeout to prevent hanging
                await asyncio.wait_for(
                    self.ws.send(json.dumps(event)), 
                    timeout=5.0  # 5 second timeout
                )
                print(f"[RealtimeDialogueSystem] Sent event: {event['type']}")
            else:
                debug_print("⚠️ WebSocket not connected - cannot send event", "WEBSOCKET_ERROR")
        except asyncio.TimeoutError:
            debug_print(f"⚠️ Timeout sending WebSocket event: {event['type']}", "WEBSOCKET_ERROR")
        except Exception as e:
            debug_print(f"❌ Error sending WebSocket event {event['type']}: {e}", "WEBSOCKET_ERROR")

    def get_instructions_for_npc(self):
        """Get system instructions based on current NPC using full AI_AGENTS data."""
        base_prompt = """You are in a 3D virtual office environment engaging in voice conversation.
Keep responses natural, conversational, and concise (2-3 sentences max).
You can hear the user's voice and should respond naturally as if having a real conversation."""

        # Get the agent data from AI_AGENTS
        if self.current_npc in AI_AGENTS:
            agent = AI_AGENTS[self.current_npc]
            
            # Build comprehensive character prompt using all AI_AGENTS data
            character_prompt = f"""
You are {agent['name']}, {agent['title']} at {agent['company']}.

PERSONALITY: {agent['personality']}

BACKGROUND: {agent['background']}

EXPERTISE: You specialize in {', '.join(agent['expertise'])}.

CONVERSATION STYLE: Be {agent['conversation_style']}.

IMPORTANT: Always stay in character as {agent['name']}. Use your expertise to provide helpful, relevant responses. 
Speak naturally and professionally while maintaining your unique personality traits."""
            
            return base_prompt + character_prompt
        
        # Fallback for unknown NPCs
        if self.current_npc == "HR":
            return f"""{base_prompt}
You are Sarah Chen, HR Director at Venture Builder AI. You're warm, professional, 
and helpful with employee matters. Speak naturally and offer practical assistance."""
        else:  # CEO
            return f"""{base_prompt}
You are Michael Chen, CEO of Venture Builder AI. You're visionary yet approachable,
passionate about venture building and AI. Share insights about the company and industry."""

    def get_voice_for_npc(self):
        """Get appropriate voice based on current NPC using AI_AGENTS definitions."""
        if self.current_npc in AI_AGENTS:
            return AI_AGENTS[self.current_npc]["voice"]
        
        # Fallback voices
        if self.current_npc == "HR":
            return "shimmer"  # Female voice for HR Director
        else:  # CEO
            return "echo"     # Male voice for CEO

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
        self.clock = pygame.time.Clock()  # For FPS

    def render_debug_overlay(self):
        fps = self.clock.get_fps()
        audio_status = 'OK' if getattr(self.dialogue.audio_handler, '_initialized', False) else 'FAIL'
        overlay_text = f"FPS: {fps:.1f} | Audio: {audio_status}"
        font = pygame.font.Font(None, 28)
        surface = font.render(overlay_text, True, (255, 255, 0))
        screen = pygame.display.get_surface()
        screen.blit(surface, (10, 10))

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
        import traceback
        running = True
        try:
            while running:
                try:
                    if self.menu.active:
                        # Menu loop
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                debug_print("[EXIT] pygame.QUIT event received. Exiting main loop.", "EXIT")
                                running = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_RETURN and time.time() - self.menu.start_time > (len(TITLE) / 15 + 1):
                                    self.menu.active = False
                                    pygame.mouse.set_visible(False)
                                    pygame.event.set_grab(True)
                                elif event.key == pygame.K_ESCAPE:
                                    debug_print("[EXIT] ESCAPE key pressed in menu. Exiting main loop.", "EXIT")
                                    running = False
                        self.menu.render()
                    else:
                        # Main game loop
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                debug_print("[EXIT] pygame.QUIT event received. Exiting main loop.", "EXIT")
                                running = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    debug_print("[EXIT] ESCAPE key pressed in game. Exiting main loop.", "EXIT")
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
                                        debug_print("[SHIFT+Q] Conversation ended, player moved out of office. App continues running.", "SHIFTQ")
                                        # DO NOT set running=False or call pygame.quit() here!
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

                        # Check NPC interactions (INSTANT, no cooldown)
                        # Check distance to HR NPC
                        dx = self.player.pos[0] - self.hr_npc.pos[0]
                        dz = self.player.pos[2] - self.hr_npc.pos[2]
                        hr_distance = math.sqrt(dx*dx + dz*dz)
                        
                        # Check distance to CEO NPC
                        dx = self.player.pos[0] - self.ceo_npc.pos[0]
                        dz = self.player.pos[2] - self.ceo_npc.pos[2]
                        ceo_distance = math.sqrt(dx*dx + dz*dz)
                        
                        if hr_distance < self.interaction_distance:
                            if not self.dialogue.active or self.dialogue.current_npc != "HR":
                                # End current conversation if talking to someone else
                                if self.dialogue.active:
                                    self.dialogue.end_conversation()
                                debug_print("👋 User approached Sarah (HR Director) - Starting conversation!", "APPROACH")
                                self.dialogue.start_conversation("HR", self.player.pos)
                        elif ceo_distance < self.interaction_distance:
                            if not self.dialogue.active or self.dialogue.current_npc != "CEO":
                                if self.dialogue.active:
                                    self.dialogue.end_conversation()
                                debug_print("👋 User approached Michael (CEO) - Starting conversation!", "APPROACH")
                                self.dialogue.start_conversation("CEO", self.player.pos)
                        else:
                            # End conversation IMMEDIATELY if you walk away from both
                            if self.dialogue.active:
                                debug_print("🚶 User walked away from all NPCs - Closing conversation immediately!", "DEPARTURE")
                                self.dialogue.end_conversation()
                                # Also stop any audio instantly
                                self.dialogue.audio_handler.stop_playback()
                                self.dialogue.is_speaking = False
                                self.dialogue.audio_playing = False
                                self.dialogue.interrupting = False
                                self.dialogue.npc_message = ""

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

                        # Render debug overlay
                        self.render_debug_overlay()
                        # Maintain 60 FPS
                        self.clock.tick(60)
                except Exception as e:
                    # More detailed error handling
                    error_msg = f"Game loop error: {str(e)}"
                    debug_print(f"❌ {error_msg}", "GAME_ERROR")
                    
                    # Log the full traceback
                    import traceback
                    traceback_str = traceback.format_exc()
                    debug_print(f"📋 Full traceback: {traceback_str}", "GAME_ERROR")
                    
                    # Try to save crash log
                    try:
                        with open("crash_log.txt", "a") as f:
                            f.write(f"Game loop crash at {time.ctime()}:\n")
                            f.write(f"Error: {error_msg}\n")
                            f.write(f"Traceback:\n{traceback_str}\n")
                            f.write("-" * 50 + "\n")
                    except Exception as log_error:
                        debug_print(f"⚠️ Could not write crash log: {log_error}", "GAME_ERROR")
                    
                    # Try to clean up dialogue system
                    try:
                        if hasattr(self, 'dialogue') and self.dialogue.active:
                            debug_print("🧹 Attempting to clean up dialogue system...", "GAME_ERROR")
                            self.dialogue.end_conversation()
                    except Exception as cleanup_error:
                        debug_print(f"⚠️ Error during dialogue cleanup: {cleanup_error}", "GAME_ERROR")
                    
                    # Continue running or exit based on error severity
                    if "CRITICAL" in str(e).upper() or "FATAL" in str(e).upper():
                        debug_print("🚨 Critical error detected - shutting down", "GAME_ERROR")
                        running = False
                    else:
                        debug_print("⚠️ Non-critical error - attempting to continue", "GAME_ERROR")
                        # Force garbage collection to clean up any corrupted data
                        import gc
                        gc.collect()
                        # Small delay to prevent rapid error loops
                        import time
                        time.sleep(0.1)
        except Exception as e:
            # Outer exception handler for critical errors
            error_msg = f"Critical game error: {str(e)}"
            debug_print(f"🚨 {error_msg}", "CRITICAL")
            
            try:
                import traceback
                traceback_str = traceback.format_exc()
                debug_print(f"📋 Critical error traceback: {traceback_str}", "CRITICAL")
                
                with open("crash_log.txt", "a") as f:
                    f.write(f"Critical crash at {time.ctime()}:\n")
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Traceback:\n{traceback_str}\n")
                    f.write("=" * 50 + "\n")
            except Exception as log_error:
                debug_print(f"⚠️ Could not write critical crash log: {log_error}", "CRITICAL")
            
            try:
                self.dialogue.end_conversation()
            except Exception as cleanup_error:
                debug_print(f"⚠️ Error during critical cleanup: {cleanup_error}", "CRITICAL")
            
            running = False
        finally:
            try:
                self.dialogue.end_conversation()
            except Exception as e:
                debug_print(f"Cleanup error: {e}", "SHUTDOWN_ERROR")
            pygame.quit()
            debug_print("[EXIT] pygame.quit() called. App is shutting down.", "EXIT")

# Display startup message with real-time debug info
debug_print("🎮 VENTURE BUILDER AI - DIGITAL EMPLOYEES GAME", "STARTUP")
debug_print("🗣️ SEAMLESS SPEECH-TO-SPEECH INTERACTION MODE", "STARTUP")
debug_print("🔧 Real-time terminal output is now ACTIVE!", "STARTUP")
debug_print("🎙️ Continuous voice activity detection enabled", "STARTUP")
debug_print("⚡ Auto-interruption system for natural conversation", "STARTUP")
debug_print("📊 All system events will be displayed with timestamps", "STARTUP")
debug_print("🎤 Enhanced audio processing for low latency", "STARTUP")
debug_print("🌐 WebSocket optimized for real-time interaction", "STARTUP")
debug_print("👥 No button presses required - just speak naturally!", "STARTUP")
debug_print("", "STARTUP")

# AI Agent Character Definitions
AI_AGENTS = {
    "HR": {
        "name": "Sarah Chen",
        "title": "HR Director",
        "company": "Venture Builder AI",
        "personality": "warm, professional, empathetic, helpful",
        "voice": "shimmer",  # Female voice - warm and professional
        "greeting": "Hello! I'm Sarah Chen, the HR Director here at Venture Builder AI. Welcome to our digital office! I'm here to help with any questions about our company culture, career opportunities, employee benefits, or workplace policies. How can I assist you today?",
        "background": "Sarah has over 10 years of experience in human resources and talent management. She's passionate about creating inclusive workplaces and helping employees grow their careers. She specializes in recruitment, employee development, and organizational culture.",
        "expertise": ["Recruitment & Hiring", "Employee Development", "Company Culture", "Benefits & Policies", "Workplace Relations", "Career Guidance"],
        "conversation_style": "friendly, supportive, detail-oriented, professional"
    },
    "CEO": {
        "name": "Michael Chen", 
        "title": "Chief Executive Officer",
        "company": "Venture Builder AI",
        "personality": "visionary, approachable, innovative, strategic",
        "voice": "echo",  # Male voice - authoritative and approachable
        "greeting": "Welcome! I'm Michael Chen, CEO and founder of Venture Builder AI. It's great to meet you in our virtual office space. I'm passionate about building the future of AI-powered businesses and venture development. I'd love to share insights about our company's vision, the venture building industry, or discuss innovation in AI. What would you like to know?",
        "background": "Michael is a serial entrepreneur and AI visionary who founded Venture Builder AI to democratize access to AI tools for startups and enterprises. He has 15+ years of experience in tech leadership and venture capital.",
        "expertise": ["AI Strategy", "Venture Building", "Business Development", "Innovation", "Leadership", "Technology Trends", "Startup Ecosystem"],
        "conversation_style": "inspiring, strategic, forward-thinking, engaging"
    }
}

# Create and run game
game = Game3D()
game.run()