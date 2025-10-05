import pygame

# Pygame ke sound mixer ko initialize karne ke liye function
def init_sound():
    """Initializes the pygame mixer."""
    print("Initializing sound manager...")
    pygame.mixer.init()
    print("Sound manager initialized.")

# Alarm sound ko load karne ke liye function
def load_alarm(sound_file_path):
    """Loads the alarm sound file."""
    try:
        print(f"Loading sound file: {sound_file_path}")
        alarm_sound = pygame.mixer.Sound(sound_file_path)
        print("Sound file loaded successfully.")
        return alarm_sound
    except pygame.error as e:
        print(f"Error loading sound file: {e}")
        return None

# Alarm play karne ke liye function
def play_alarm(sound_object):
    """Plays the alarm sound if it's not already playing."""
    # Check karte hain ki koi sound pehle se play to nahi ho raha
    if not pygame.mixer.get_busy():
        sound_object.play(-1)  # -1 ka matlab hai loop mein play karte raho

# Alarm stop karne ke liye function
def stop_alarm():
    """Stops any currently playing sound."""
    pygame.mixer.stop()
