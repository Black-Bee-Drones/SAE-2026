import yasmin
import sys
import tty
import termios
import json
import os
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from ..parameters import SIMULATION
from zaxis.telemetry.mavlink import MavlinkConnection

from ..parameters import LOCATIONS

from zaxis.drone import Drone

class Init(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT, "GROUND_MONITOR"])
        self.coords_file = os.path.expanduser("~/.faulty_or_not_mission_coords.json")

    def save_debug_coordinates(self, locations):
        """Save coordinates to a file for persistence"""
        try:
            with open(self.coords_file, 'w') as f:
                json.dump(locations, f)
        except Exception as e:
            yasmin.YASMIN_LOG_WARN(f"Failed to save coordinates: {e}")

    def load_debug_coordinates(self):
        """Load previously saved coordinates"""
        try:
            if os.path.exists(self.coords_file):
                with open(self.coords_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            yasmin.YASMIN_LOG_WARN(f"Failed to load saved coordinates: {e}")
        return None

    def get_key_input(self):
        """Capture a pressed key without needing to press Enter"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def display_menu(self, selected_index):
        """Display menu with selected item highlighted"""

        os.system('clear')
        
        print("=" * 50)
        print("       Faulty or Not - SAE Eletroquad 2026")
        print("=" * 50)
        print("\nSelect an option:")
        print("Use ↑/↓ to navigate, Enter to confirm\n")
        
        options = ["Run Mission", "Run Ground Monitor", "Edit Coordinates"]
        
        for i, option in enumerate(options):
            if i == selected_index:
                print(f"  ▶ {option} ◀")
            else:
                print(f"    {option}")
        
        print(f"\nSelected option: {options[selected_index]}")

    def select_mode(self):
        """Allow user to select option using keyboard arrows"""
        selected_index = 0
        options = ["Run Mission", "Run Ground Monitor", "Edit Coordinates"]
        
        while True:
            self.display_menu(selected_index)
            
            key = self.get_key_input()
            
            if key == '\x1b[A':  # Up arrow
                selected_index = (selected_index - 1) % len(options)
            elif key == '\x1b[B':  # Down arrow
                selected_index = (selected_index + 1) % len(options)
            elif key == '\r' or key == '\n': 
                return options[selected_index]
            elif key == '\x03': 
                raise KeyboardInterrupt("Menu cancelled by user")

    def edit_coordinates(self):
        """Allow editing saved coordinates in JSON"""
        os.system('clear')
        print("=" * 50)
        print("              COORDINATE EDITOR")
        print("=" * 50)
        
        # Load current coordinates
        current_coords = self.load_debug_coordinates()
        if not current_coords:
            current_coords = list(LOCATIONS)  # Use default if no saved coordinates
        
        print(f"\nCurrent coordinates: {current_coords}")
        
        print("\nOptions:")
        print("1. Reset to default coordinates")
        print("2. Keep current coordinates")
        print("3. Insert new coordinates")
        
        choice = input("\nChoose (1-3): ").strip()
        
        if choice == '1':
            new_coords = list(LOCATIONS)
        elif choice == '2':
            new_coords = current_coords
        elif choice == '3':
            print("\nEnter new coordinates (format: x,y,z)")
            print("Type 'done' when finished\n")
            new_coords = []
            i = 1
            while True:
                coord_input = input(f"Waypoint {i}: ").strip()
                if coord_input.lower() == 'done':
                    break
                try:
                    coords = coord_input.split(',')
                    if len(coords) != 3:
                        print("Invalid format! Use: x,y,z")
                        continue
                    x, y, z = float(coords[0].strip()), float(coords[1].strip()), float(coords[2].strip())
                    new_coords.append((x, y, z))
                    print(f"Waypoint {i} added: ({x}, {y}, {z})")
                    i += 1
                except ValueError:
                    print("Invalid coordinates! Use valid numbers.")
            
            if not new_coords:
                print("No coordinates entered, using default.")
                new_coords = list(LOCATIONS)
        else:
            print("Invalid option, keeping current coordinates.")
            new_coords = current_coords
        
        # Save new coordinates
        self.save_debug_coordinates(new_coords)
        
        print(f"\nCoordinates saved: {new_coords}")
        input("\nPress Enter to continue...")
        return new_coords

    def get_mission_coordinates(self):
        """Load coordinates for mission (from JSON or default)"""
        saved_coords = self.load_debug_coordinates()
        if saved_coords:
            print(f"Using saved coordinates: {saved_coords}")
            return saved_coords
        else:
            print(f"Using default coordinates: {LOCATIONS}")
            return LOCATIONS

    def execute(self, blackboard : Blackboard):
        try:
            yasmin.YASMIN_LOG_INFO("Faulty or Not - SAE Eletroquad 2026")

            # Interactive selection menu
            while(True):
                selected_option = self.select_mode()
                
                if selected_option == "Edit Coordinates":
                    self.edit_coordinates()
                elif selected_option == "Run Ground Monitor":
                    yasmin.YASMIN_LOG_INFO("Ground Monitor mode selected.")
                    return "GROUND_MONITOR"
                else:
                    break
                
                
            # Run mission
            os.system('clear')
            locations = self.get_mission_coordinates()
            yasmin.YASMIN_LOG_INFO("Mission started.")
            
            blackboard["control_index"] = 0
            blackboard["locations"] = locations

            self.drone = Drone(MavlinkConnection())
            self.drone.connection.connect("udpin:0.0.0.0:14550", 921600)
            blackboard["drone"] = self.drone
            
            self.drone.set_mode(self.drone.FlightMode.GUIDED).wait()

            return SUCCEED
        except KeyboardInterrupt:
            yasmin.YASMIN_LOG_WARN("Initialization cancelled by user")
            return ABORT
        except Exception as e:
            import traceback
            print("Failed start mission: ", traceback.format_exc())
            raise