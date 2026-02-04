# Connect Pro Controller via Bluetooth to Jetson Orin

Steps to pair a Nintendo Switch Pro Controller over Bluetooth with a Jetson Orin.

## 1. Put the Pro Controller in pairing mode

- Locate the **Sync** button (small circular button near the USB-C port on top of the controller).
- Press and hold it for **3–5 seconds** until the **four LED indicators** at the bottom start **flashing rapidly**. That’s pairing mode.

## 2. Pair on the Jetson

### Option A: Using `bluetoothctl` (terminal)

1. **Start Bluetooth and open the Bluetooth control tool:**
   ```bash
   sudo systemctl start bluetooth
   bluetoothctl
   ```

2. **Inside `bluetoothctl`, run:**
   ```
   power on
   agent on
   default-agent
   scan on
   ```
   Wait until you see a device named **"Pro Controller"** (or similar Nintendo name) and note its **MAC address** (e.g. `XX:XX:XX:XX:XX:XX`).

3. **Stop scanning, then pair, trust, and connect:**
   ```
   scan off
   pair XX:XX:XX:XX:XX:XX
   trust XX:XX:XX:XX:XX:XX
   connect XX:XX:XX:XX:XX:XX
   ```

4. **Exit:**
   ```
   quit
   ```

### Option B: Using the GUI

If you have a desktop environment (e.g. Ubuntu on Jetson):

- Open **Settings → Bluetooth**, ensure Bluetooth is **On**, then **search for devices**.
- When **"Pro Controller"** (or similar) appears, select it and choose **Pair** / **Connect**.

## 3. Verify the controller is recognized

- **List input devices:**
  ```bash
  ls /dev/input/
  ```
  You should see entries like `js0` (and possibly `event*`).

- **Test with `jstest` (if installed):**
  ```bash
  sudo apt install joystick
  jstest /dev/input/js0
  ```
  Moving sticks and pressing buttons should show changing values.

- **Or with `evtest`:**
  ```bash
  sudo apt install evtest
  sudo evtest
  ```
  Select the Pro Controller from the list and move/press controls to see events.

## 4. Auto-start Bluetooth on boot (optional)

```bash
sudo systemctl enable bluetooth
```

## 5. Reconnecting later

- Turn on the Pro Controller (e.g. press **Home**).
- It should reconnect automatically if it was previously trusted. If not, run `bluetoothctl` again and use `connect XX:XX:XX:XX:XX:XX` with the same MAC address.

## Troubleshooting

If Bluetooth shows "connected" (e.g. `[Pro Controller]#` in bluetoothctl) but the controller does not work as an input device, work through the following.

### (1) In bluetoothctl: check services and connect HID

The controller may be connected at the link level but not with the **HID** (Human Interface Device) profile, so no gamepad input is exposed.

1. With the Pro Controller on and in range, run:
   ```
   devices
   ```
   Note the **MAC address** of "Pro Controller".

2. Connect explicitly:
   ```
   connect XX:XX:XX:XX:XX:XX
   ```
   (use your MAC). Wait for "Connection successful" or "Failed".

3. Check device info:
   ```
   info XX:XX:XX:XX:XX:XX
   ```
   Look for **Connected: yes** and, under **Services** / **UUIDs**, something like **Human Interface Device** or **HID**. If Connected is yes but there is no HID-related service, the kernel may not support this controller over Bluetooth (e.g. missing `hid_nintendo`).

### (2) See if the kernel creates an input device

1. In a separate terminal, run:
   ```bash
   ls /dev/input/
   ```
2. In bluetoothctl, run **`connect XX:XX:XX:XX:XX:XX`** again.
3. After a few seconds, run:
   ```bash
   ls /dev/input/
   dmesg | tail -20
   ```
   If a new **eventX** (or **js0** after `sudo modprobe joydev`) appears when you connect, the controller is working at the kernel level. If **no** new device appears, run:
   ```bash
   modinfo hid_nintendo
   lsmod | grep hid
   ```
   If **hid_nintendo** is not found or not loaded, the Jetson kernel likely does not support the Pro Controller over Bluetooth; use a kernel with **CONFIG_HID_NINTENDO** or connect the controller via **USB** and check `/dev/input/` again.

### (3) Trust and reconnect

In bluetoothctl, try:

```
trust XX:XX:XX:XX:XX:XX
disconnect XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
```

Then re-check `/dev/input/` and `dmesg`. If there is still no new **event*** or **js***, the blocker is almost certainly missing **hid_nintendo** (or equivalent) support in the kernel, not bluetoothctl.

Tip: When testing with Jetson Orin, I have sometime seen issues reconnecting bluetooth. I have got around this by doing a `remove XX:XX:XX:XX:XX:XX` and `pair XX:XX:XX:XX:XX:XX` again that has resolved this issue.

---

## Notes

- **Permissions:** If your app reads from `/dev/input/js0` (or `event*`) directly, it may need to run as root or your user may need to be in the `input` group:  
  `sudo usermod -aG input $USER` (then log out and back in).

