## IMPORTANT:
- Usage of this program inside a virtual environment is STRICTLY encouraged!
- This program is designed to work with Python version 3.11. Usage of newer Python versions will create conflicts due to the dependencies used in this program support different Python versions.
- If you run this program on Raspberry Pi, ensure the device runs on 64-bit OS.

## USAGE:
This program has 2 different sets of dependency installation instructions:
    - one for usage in VSCode (Windows / MacOS) `req_vscode.txt`
    - one for usage in Raspberry Pi OS `req_rpi4b.txt`

### Instruction for VSCode on Windows:
Step 1 - create virtual environment: `py -3.11 -m venv [your_virtual_env_name]`
    -> Activate virtual environment: `[your_virtual_env_name]\Scripts\activate`
Step 2 - install dependencies:
```
    pip install --upgrade pip
    pip install -r req_vscode.txt
```

### Instruction for Raspberry Pi:
Step 1 - ensure Python 3.11 is available in your system: `python3.11 --version`
    -> If not available, install it: `sudo apt-get install python3.11 python3.11-venv`
Step 2 - create virtual environment:
```
    cd /path/to/your/project  # Navigate to your project directory
    python3.11 -m venv venv_trash_classifier
    source venv_trash_classifier/bin/activate
```
Optional - install system-level dependencies for Pygame (if not available on system):
```
    sudo apt-get update
    sudo apt-get install libsdl2-mixer-2.0-0
```
Step 3 - install Python dependencies:
```
    pip install --upgrade pip
    pip install -r req_rpi4b.txt
```

### Potential challenges on Raspberry Pi & Troubleshooting:
- `tensorflow-io-gcs-filesystem`: While `0.34.0` is compatible with Python 3.11 and TensorFlow 2.13.0, finding a pre-built wheel for `aarch64` might still be an issue for some users or specific Raspberry Pi OS versions.
    * If `pip` fails with "No matching distribution found", you might need to try slightly older compatible versions of `tensorflow-io-gcs-filesystem` (e.g., `0.33.0` or even `0.32.0`) in case a specific wheel is missing for `0.34.0` on your exact RPi setup.
    * As a last resort for `tensorflow-io-gcs-filesystem` on RPi, you might have to compile it from source, which is a lengthy and resource-intensive process. (This is generally outside the scope of a `req_rpi4b.txt` but good to be aware of).
- Other packages: Most other packages like `opencv-python`, `numpy`, `Pillow`, `pygame` generally have reliable `aarch64` wheels for Python 3.11.
