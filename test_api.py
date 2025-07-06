#!/usr/bin/env python3
"""
OpenAI API Connection Test Script
This script tests your OpenAI API key and connection to help diagnose issues.
"""

import os
import sys
import asyncio
import websockets
import json
from dotenv import load_dotenv
from openai import OpenAI
import requests
import time

def test_basic_connectivity():
    """Test basic internet connectivity"""
    print("🌐 Testing basic internet connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connection: OK")
            return True
        else:
            print(f"❌ Internet connection issue: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Internet connection failed: {e}")
        return False

def test_openai_api_key():
    """Test OpenAI API key with regular API"""
    print("\n🔑 Testing OpenAI API key...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No API key found in .env file")
        print("📋 Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return False, None
    
    print(f"🔍 API key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API key: VALID")
        print(f"📝 Test response: {response.choices[0].message.content}")
        return True, api_key
        
    except Exception as e:
        print(f"❌ OpenAI API key test failed: {e}")
        return False, api_key

async def test_realtime_api(api_key):
    """Test OpenAI Realtime API WebSocket connection"""
    print("\n🔌 Testing OpenAI Realtime API connection...")
    
    url = "wss://api.openai.com/v1/realtime"
    model = "gpt-4o-realtime-preview"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"🔗 Connecting to: {url}")
        print(f"🤖 Model: {model}")
        
        # Try with a shorter timeout first
        ws = await asyncio.wait_for(
            websockets.connect(
                f"{url}?model={model}",
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            ),
            timeout=10  # 10 second timeout
        )
        
        print("✅ Realtime API connection: SUCCESS")
        await ws.close()
        return True
        
    except asyncio.TimeoutError:
        print("❌ Realtime API connection: TIMEOUT")
        print("🔍 This could mean:")
        print("   - Your API key doesn't have Realtime API access")
        print("   - Network/firewall blocking the connection")
        print("   - OpenAI Realtime API is experiencing issues")
        return False
        
    except Exception as e:
        print(f"❌ Realtime API connection failed: {e}")
        return False

def test_audio_system():
    """Test audio system (PyAudio)"""
    print("\n🎤 Testing audio system...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    input_devices.append(dev_info['name'])
            except:
                continue
        
        if input_devices:
            print(f"✅ Audio input devices found: {len(input_devices)}")
            print(f"🎤 Available microphones: {', '.join(input_devices[:3])}")
        else:
            print("❌ No audio input devices found")
            
        p.terminate()
        return len(input_devices) > 0
        
    except Exception as e:
        print(f"❌ Audio system test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 OpenAI API Connection Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    internet_ok = test_basic_connectivity()
    
    # Test 2: OpenAI API key
    api_ok, api_key = test_openai_api_key()
    
    # Test 3: Realtime API (only if API key works)
    realtime_ok = False
    if api_ok and api_key:
        realtime_ok = await test_realtime_api(api_key)
    
    # Test 4: Audio system
    audio_ok = test_audio_system()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    print(f"🌐 Internet Connection: {'✅ OK' if internet_ok else '❌ FAILED'}")
    print(f"🔑 OpenAI API Key: {'✅ VALID' if api_ok else '❌ INVALID'}")
    print(f"🔌 Realtime API: {'✅ ACCESSIBLE' if realtime_ok else '❌ NOT ACCESSIBLE'}")
    print(f"🎤 Audio System: {'✅ WORKING' if audio_ok else '❌ ISSUES'}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    if not internet_ok:
        print("🌐 Fix your internet connection first")
    elif not api_ok:
        print("🔑 Check your OpenAI API key in the .env file")
        print("   - Get your key from: https://platform.openai.com/api-keys")
        print("   - Create .env file with: OPENAI_API_KEY=your_key_here")
    elif not realtime_ok:
        print("🔌 Realtime API not accessible. Options:")
        print("   1. Request access to Realtime API beta from OpenAI")
        print("   2. Use fallback mode (regular API + TTS)")
        print("   3. Check if your account has sufficient credits")
    elif not audio_ok:
        print("🎤 Audio system needs attention:")
        print("   - Check microphone permissions")
        print("   - Ensure microphone is connected")
    else:
        print("✅ All systems should work! Try running the main app again.")

if __name__ == "__main__":
    asyncio.run(main()) 