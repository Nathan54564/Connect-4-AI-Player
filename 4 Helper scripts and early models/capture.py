import serial
import time

# ==========================================
# CHANGE THIS TO YOUR PORT (e.g., COM3, /dev/ttyUSB0)
# ==========================================
SERIAL_PORT = 'COM15' 
BAUD_RATE = 115200

def get_image():
    try:
        # Open Serial Connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Connected to {SERIAL_PORT}...")
        time.sleep(2) # Give board time to reset if needed

# Send 'g' command to Arduino to trigger the photo
        print("Requesting photo...")
        ser.write(b'g')
        ser.flush()

        # Wait for the Start Signal
        print("Waiting for response (Press Ctrl+C to stop)...")
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # === NEW: Print what the board is saying ===
            if line: 
                print(f"Board says: {line}") 
            # ===========================================

            if "IMAGESTART" in line:
                break
            
        # Read the file size
        size_str = ser.readline().decode('utf-8').strip()
        image_size = int(size_str)
        print(f"Receiving image ({image_size} bytes)...")

        # Read the binary data
        image_data = ser.read(image_size)

        # Save to file
        filename = f"captured_image_{int(time.time())}.jpg"
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        print(f"Success! Saved as {filename}")
        ser.close()

    except Exception as e:
        print(f"Error: {e}")
        # Hint: If you get Access Denied, close the Arduino Serial Monitor!

if __name__ == "__main__":
    get_image()