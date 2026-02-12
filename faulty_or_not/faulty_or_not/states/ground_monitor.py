from std_msgs.msg import UInt8
from yasmin import Blackboard, State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED
import sys
import tty
import termios
import os
import json
from playsound import playsound
from ament_index_python.packages import get_package_share_directory


class GroundMonitor(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED])
        self.node = YasminNode.get_instance()
        self.waypoints_file = os.path.expanduser("~/.faulty_or_not_waypoints.json")

        self.waypoints = []  # Will be set after user selects waypoints
        
        # Map gauge class to actual pressure value
        self.class_to_pressure = {
            0: 20,
            1: 40,
            2: 60,
            3: 80,
            4: 100,
            5: 120
        }


        self.node.get_logger().info("GroundMonitor initialized")

    def gauge_callback(self, msg: UInt8):
        # Extract packed data: (gauge_reading << 5) | (control_index & 0x1F)
        packed_data = msg.data
        
        # Extract control_index (lower 5 bits)
        control_index = packed_data & 0x1F
        
        # Extract gauge_reading (upper 3 bits)
        gauge_reading = (packed_data >> 5) & 0x07
        
        self.node.get_logger().info(f"Received - Control Index: {control_index}, Gauge Class: {gauge_reading}")
        
        # Check if we have waypoints set
        if not self.waypoints or len(self.waypoints) == 0:
            self.node.get_logger().warn("Waypoints not set yet!")
            return
        
        # Check if control_index is valid
        if control_index >= len(self.waypoints):
            self.node.get_logger().error(f"Control index {control_index} out of bounds! Only {len(self.waypoints)} waypoints set.")
            return
        
        # Get expected pressure from waypoints
        expected_pressure = int(self.waypoints[control_index])
        
        actual_pressure = self.class_to_pressure[gauge_reading]
        
        # Check if gauge reading is lower than expected (FAULTY)
        is_faulty = actual_pressure > expected_pressure
        
        self.node.get_logger().info(
            f"Waypoint {control_index}: Expected={expected_pressure} psi, "
            f"Actual={actual_pressure} psi, Status={'FAULTY' if is_faulty else 'OK'}"
        )
        
        package_share_directory = get_package_share_directory('faulty_or_not')
        # Play sound alert if faulty
        if is_faulty:
            path = os.path.join(package_share_directory, 'assets', 'alarm.mp3')
        else:
            path = os.path.join(package_share_directory, 'assets', 'not_faulty.mp3')

        playsound(path)

    def save_waypoints(self, waypoints):
        """Save waypoints to a file for persistence"""
        try:
            with open(self.waypoints_file, 'w') as f:
                json.dump(waypoints, f)
            self.node.get_logger().info(f"Waypoints saved to {self.waypoints_file}")
        except Exception as e:
            self.node.get_logger().warn(f"Failed to save waypoints: {e}")

    def load_waypoints(self):
        """Load previously saved waypoints"""
        try:
            if os.path.exists(self.waypoints_file):
                with open(self.waypoints_file, 'r') as f:
                    waypoints = json.load(f)
                    self.node.get_logger().info(f"Loaded saved waypoints: {waypoints}")
                    return waypoints
        except Exception as e:
            self.node.get_logger().warn(f"Failed to load saved waypoints: {e}")
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

    def display_waypoint_menu(self, waypoint_num, selected_index):
        """Display menu for selecting waypoint value"""
        os.system('clear')
        
        print("=" * 50)
        print("       Faulty or Not - SAE Eletroquad 2026")
        print("=" * 50)
        print(f"\nWaypoint {waypoint_num}")
        print("Use ←/→ to navigate, Enter to confirm\n")
        
        options = ["0", "20", "40", "60", "80", "100"]
        
        # Show current selection with arrows
        print(f"◀ {options[selected_index]} ▶")
        print()

    def confirm_waypoints(self, waypoints, selected_index):
        """Ask user to confirm waypoints or add more"""
        os.system('clear')
        
        print("=" * 50)
        print("       Faulty or Not - SAE Eletroquad 2026")
        print("=" * 50)
        print("\nSelected waypoints:\n")
        
        for i, wp in enumerate(waypoints, 1):
            print(f"  Waypoint {i}: {wp}")
        
        print("\n" + "=" * 50)
        print("\nWhat would you like to do?")
        print("Use ↑/↓ to navigate, Enter to confirm\n")
        
        options = ["Add another waypoint", "Finish and continue", "Clear and start over"]
        
        for i, option in enumerate(options):
            if i == selected_index:
                print(f"  ▶ {option} ◀")
            else:
                print(f"    {option}")
    
    def ask_continue_adding(self, waypoints):
        """Ask user if they want to add another waypoint or finish"""
        selected_index = 0
        
        while True:
            self.confirm_waypoints(waypoints, selected_index)
            
            key = self.get_key_input()
            
            if key == '\x1b[A':  # Up arrow
                selected_index = (selected_index - 1) % 3
            elif key == '\x1b[B':  # Down arrow
                selected_index = (selected_index + 1) % 3
            elif key == '\r' or key == '\n':  # Enter
                if selected_index == 0:  # Add another waypoint
                    return 'add'
                elif selected_index == 1:  # Finish and continue
                    return 'finish'
                else:  # Clear and start over
                    return 'clear'
            elif key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt("Menu cancelled by user")

    def display_confirmation_menu_old(self, waypoints, selected_index):
        """Display confirmation menu with saved waypoints (for loaded waypoints)"""
        os.system('clear')
        
        print("=" * 50)
        print("       Faulty or Not - SAE Eletroquad 2026")
        print("=" * 50)
        print("\nPreviously saved waypoints:\n")
        
        for i, wp in enumerate(waypoints, 1):
            print(f"  Waypoint {i}: {wp}")
        
        print("\n" + "=" * 50)
        print("\nUse these waypoints?")
        print("Use ↑/↓ to navigate, Enter to confirm\n")
        
        options = ["Yes, continue", "No, select again"]
        
        for i, option in enumerate(options):
            if i == selected_index:
                print(f"  ▶ {option} ◀")
            else:
                print(f"    {option}")

    def confirm_saved_waypoints(self, waypoints):
        """Ask user to confirm previously saved waypoints"""
        selected_index = 0
        
        while True:
            self.display_confirmation_menu_old(waypoints, selected_index)
            
            key = self.get_key_input()
            
            if key == '\x1b[A':  # Up arrow
                selected_index = (selected_index - 1) % 2
            elif key == '\x1b[B':  # Down arrow
                selected_index = (selected_index + 1) % 2
            elif key == '\r' or key == '\n':  # Enter
                return selected_index == 0  # True if "Yes", False if "No"
            elif key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt("Menu cancelled by user")

    def select_waypoints(self):
        """Allow user to select waypoint values using keyboard arrows"""
        # Check if there are saved waypoints
        saved_waypoints = self.load_waypoints()
        
        # Use saved waypoints as default if available
        if saved_waypoints and len(saved_waypoints) > 0:
            if self.confirm_saved_waypoints(saved_waypoints):
                return saved_waypoints
        
        # Manual waypoint selection
        while True:
            options = ["0", "20", "40", "60", "80", "100"]
            waypoints = []
            
            # Keep adding waypoints until user chooses to finish
            waypoint_num = 1
            while True:
                selected_index = 0  # Start at first option for each waypoint
                
                # Select waypoint value
                while True:
                    self.display_waypoint_menu(waypoint_num, selected_index)
                    
                    key = self.get_key_input()
                    
                    if key == '\x1b[D':  # Left arrow
                        selected_index = (selected_index - 1) % len(options)
                    elif key == '\x1b[C':  # Right arrow
                        selected_index = (selected_index + 1) % len(options)
                    elif key == '\r' or key == '\n':  # Enter
                        waypoints.append(options[selected_index])
                        self.node.get_logger().info(f"Waypoint {waypoint_num} saved: {options[selected_index]}")
                        break  # Move to confirmation
                    elif key == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt("Menu cancelled by user")
                
                # After adding a waypoint, ask what to do next
                action = self.ask_continue_adding(waypoints)
                
                if action == 'add':
                    waypoint_num += 1
                    continue  # Add another waypoint
                elif action == 'finish':
                    if len(waypoints) == 0:
                        self.node.get_logger().warn("You must add at least one waypoint!")
                        continue
                    return waypoints  # All done!
                elif action == 'clear':
                    break  # Break inner loop to restart
            
            # If we get here, user chose to clear and restart
            # The outer while True will restart the process


    def execute(self, blackboard: Blackboard):
        # Get waypoints from user
        waypoints = self.select_waypoints()
        
        # Save waypoints to file
        self.save_waypoints(waypoints)
        
        # Save waypoints to blackboard and instance variable
        blackboard.waypoints = waypoints
        self.waypoints = waypoints  # Set instance variable for callbacks to use
        
        self.node.get_logger().info(f"All waypoints saved: {waypoints}")
        self.node.get_logger().info(f"Monitoring {len(waypoints)} waypoints for faulty gauges...")
        self.node.get_logger().info("Press Ctrl+C to stop monitoring and exit")

        self.node.create_subscription(
            UInt8,
            "/gauge/reading",
            self.gauge_callback,
            10
        )

        # Keep monitoring until Ctrl+C
        try:
            import time
            while True:
                time.sleep(0.1)  # Sleep to avoid busy waiting
        except KeyboardInterrupt:
            self.node.get_logger().info("Monitoring stopped by user")
        
        return SUCCEED