import asyncio
import sys
import time
# import cv2
import os
import keyboard


from bleak import BleakScanner, BleakClient

# BLE configuration
DEVICE_NAME = "EAntenna"
SERVICE_UUID = "00001111-0000-1000-8000-00805F9B34FB"
NOTIFICATION_CHARACTERISTIC_UUID = "00002222-0000-1000-8000-00805F9B34FB"
RECEIVE_CHARACTERISTIC_UUID = "00003333-0000-1000-8000-00805F9B34FB"

# Output file to save sensing data
OUTPUT_FOLDER = "xxx"
start_time = None
time_tmp = time.time()
end_time = None

#set sensing threshold 
LOWER_LIMIT = -1000
UPPER_LIMIT = 1000

# Transmission speed count variables
SAMPLE_INTERVAL = 3  # Time interval in seconds
MAX_PACKETS_PER_WRITE = 1000  # Number of data samples to write in each file flush
sample_count = 0
sample_count_update_message = 0
COUNT_LIMIT_UPDATE_MESSAGE = 50

# Buffer for storing sensing data
data_buffer = []

label = 1



async def handle_data(sender: int, data: bytearray):
    global Bx1, sample_count, client, start_time, end_time, time_tmp, sample_count_update_message  # Declare as global

    if start_time is None:
        start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    data_str = data.decode("utf-8")
    [Bx, BY, BZ, Input,Output,speed0, speed1] = data_str.split('\t')
    print(f"Sensing Data: {data_str}")
    data_withLable = [Bx, BY, BZ, Input,Output,speed0, speed1];
    data_str = "\t".join(str(e) for e in data_withLable)
    if ((float(Bx) > LOWER_LIMIT) and (float(Bx) < UPPER_LIMIT)):
        # Save the sensing data to the buffer
        data_buffer.append(data_str)
        sample_count += 1
        sample_count_update_message += 1
    # Check if time interval has elapsed
    current_time = time.time()
    elapsed_time = current_time - time_tmp

    if elapsed_time >= SAMPLE_INTERVAL:
        # Calculate transmission speed
        transmission_speed = sample_count / elapsed_time
        print(f"Transmission Speed: {transmission_speed} samples/second")

        # Reset sample count and start time
        sample_count = 0
        time_tmp = current_time
 

async def send_message(client, message):
    await client.write_gatt_char(RECEIVE_CHARACTERISTIC_UUID, message.encode(), response=True)
    # print(f"Sent Message: {message}")


async def scan_and_connect():
    global client, value, Bx1
    scanner = BleakScanner()
    devices = await scanner.discover()

    for device in devices:
        print(f"Discovered Device: {device.name} ({device.address})")

    device = next((d for d in devices if d.name == DEVICE_NAME), None)

    if device is None:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    async with BleakClient(device) as client:
        await client.is_connected()
        print(f"Connected to device: {device.name} ({device.address})")

        # Start receiving notifications for the characteristic
        await client.start_notify(NOTIFICATION_CHARACTERISTIC_UUID, handle_data)

        # Wait for 'q' keypress to quit the reading process
        print("Press 'q' to quit...")

        while True:
            await asyncio.sleep(0.02)  # Check for keypress more frequently
            # Check if 'q' key is pressed
            if keyboard.is_pressed('q'):
                await client.disconnect()
                flush_buffer_to_file()
                break
            elif keyboard.is_pressed('a'):
                await send_message(client, "120")
  

def flush_buffer_to_file():
    global data_buffer, sample_count, start_time, end_time

    if end_time is None:
        end_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Create the output file name based on start time and end time
    output_file = f"{label}_To_{start_time}_To_{end_time}.txt"
    output_path = os.path.join(OUTPUT_FOLDER, output_file)

    # Open the output file in append mode
    with open(output_path, "a") as file:
        # Write the contents of the buffer to the file
        file.write("\n".join(data_buffer))

        # Flush the file to ensure data is written
        file.flush()

        # Clear the buffer and reset the sample count
        data_buffer = []
        sample_count = 0

def main():
    print("Scanning for Bluetooth devices...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scan_and_connect())
    flush_buffer_to_file()

if __name__ == "__main__":
    # Initialize OpenCV window for capturing keypress
    # cv2.namedWindow("Window")
    main()
    # cv2.destroyAllWindows()
