from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

import rclpy

from faulty_or_not.states import (
    Init, 
    Takeoff,
    Navigation,
    GaugeReading,
    AudioFeedback,
    ReturnToLaunch,
    GroundMonitor
)
from faulty_or_not.parameters import MODEL_PATH


class FaultyOrNot(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT, "END"])
        self.add_state(
            "INIT",
            Init(),
            transitions={SUCCEED:"TAKEOFF", "GROUND_MONITOR":"GROUND_MONITOR",ABORT:ABORT},
        )
        self.add_state(
            "GROUND_MONITOR",
            GroundMonitor(),
            transitions={SUCCEED:SUCCEED}
        )
        self.add_state(
            "TAKEOFF",
            Takeoff(),
            transitions={SUCCEED:"NAVIGATION", ABORT:"RETURN_TO_LAUNCH"},
        )
        self.add_state(
            "NAVIGATION",
            Navigation(),
            transitions={SUCCEED:"GAUGE_READING", ABORT:"RETURN_TO_LAUNCH"},
        )
        self.add_state(
            "GAUGE_READING",
            GaugeReading(model_path=MODEL_PATH),
            transitions={SUCCEED:"AUDIO_FEEDBACK", ABORT:"RETURN_TO_LAUNCH"},
        )
        self.add_state(
            "AUDIO_FEEDBACK",
            AudioFeedback(),
            transitions={SUCCEED:"GAUGE_READING", "END":"RETURN_TO_LAUNCH"},
        )
        self.add_state(
            "RETURN_TO_LAUNCH",
            ReturnToLaunch(),
            transitions={SUCCEED:"END", ABORT:"END"}
        )

        self.set_start_state("INIT")


def main():
    # Initialize ROS 2
    rclpy.init()
    
    try:
        # Create the state machine
        sm = FaultyOrNot()
        
        # Execute the state machine
        outcome = sm()
        
        print(f"State machine finished with outcome: {outcome}")
        
    except KeyboardInterrupt:
        print("State machine interrupted by user")
    except Exception as e:
        print(f"State machine failed with error: {e}")
    finally:
        # Shutdown yasmin node and ROS 2
        rclpy.shutdown()


if __name__ == "__main__":
    main()