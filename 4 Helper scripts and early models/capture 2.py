import serial
import time

port = "COM13"      # change to your Arduino port
baud = 115200       # match your Arduino baud rate
outfile = "log.txt"

ser = serial.Serial(port, baud, timeout=1)

with open(outfile, "w") as f:
    while True:
        try:
            line = ser.readline().decode(errors="ignore")
            if line:
                print(line, end="")
                f.write(line)
        except KeyboardInterrupt:
            print("\nStopped.")
            break