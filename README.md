## Arduino Firmware Development and Updates

The Arduino firmware lives in:

arduino/behavioral_controller/

This code does NOT update itself automatically. All changes must be pushed
manually by a developer machine.

### Editing the Arduino code
1. Open:
   arduino/behavioral_controller/behavioral_controller.ino
   in the Arduino IDE.
2. Make and test changes locally.
3. Upload to the Arduino over USB as usual.

### Pushing firmware changes to GitHub
After verifying the firmware works:

1. Open a terminal in the repository root:
   behavior-control/
2. Run:
   git status
   git add arduino/behavioral_controller
   git commit -m "Update Arduino firmware: <brief description>"
   git push

### Pulling firmware changes on another PC
On any machine that needs the latest firmware source:

git pull

Then upload the firmware to the Arduino using the Arduino IDE.

### Important rules
- Do NOT edit Arduino files directly on runtime-only PCs.
- Do NOT push untested firmware.
- Firmware versioning is handled by Git commit history.
