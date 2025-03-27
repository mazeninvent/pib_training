import os
from imitation.algorithms.bc import reconstruct_policy
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_hat import BrickHAT
from tinkerforge.bricklet_servo_v2 import BrickletServoV2
import time

HOST = "localhost"
PORT = 4223
UIDservo1 = '29F5'
UIDservo2 = '29F3'

ipcon = IPConnection()
ipcon.connect(HOST, PORT)
servoBrick1 = BrickletServoV2(UIDservo1, ipcon)
servoBrick2 = BrickletServoV2(UIDservo2, ipcon)

policy = reconstruct_policy(
    os.path.join("model.zip"),
    device="cpu",
)

number_of_actions_to_generate = 500

for x in range(number_of_actions_to_generate):
    actions, _ = policy.predict(obs)
    denorm_actions = actions * 9000
    # Shoulder vertical
    servoBrick1.set_pulse_width(9, 700, 2500)
    servoBrick1.set_position(9, denorm_actions[0])
    servoBrick1.set_motion_configuration(9, 9000, 9000, 9000)
    servoBrick1.set_enable(9, True)
    # Upper arm rotation
    servoBrick2.set_pulse_width(9, 700, 2500)
    servoBrick2.set_position(9, denorm_actions[1])
    servoBrick2.set_motion_configuration(9, 9000, 9000, 9000)
    servoBrick2.set_enable(9, True)
    # Thumb stretch angle_thumb
    servoBrick2.set_pulse_width(1, 700, 2500)
    servoBrick2.set_position(1, denorm_actions[2])
    servoBrick2.set_enable(1, True)
    # Thumb opposition angle_thumb2
    servoBrick2.set_pulse_width(0, 700, 2500)
    servoBrick2.set_position(0, denorm_actions[3])
    servoBrick2.set_enable(0, True)
    # Index finger angle_idx
    servoBrick2.set_pulse_width(2, 700, 2500)
    servoBrick2.set_position(2, denorm_actions[4])
    servoBrick2.set_enable(2, True)
    # Middle finger angle_mid
    servoBrick2.set_pulse_width(3, 700, 2500)
    servoBrick2.set_position(3, denorm_actions[5])
    servoBrick2.set_enable(3, True)
    # Ring finger angle_rng
    servoBrick2.set_pulse_width(4, 700, 2500)
    servoBrick2.set_position(4, denorm_actions[6])
    servoBrick2.set_enable(4, True)
    # Small finger angle_ltl
    servoBrick2.set_pulse_width(5, 700, 2500)
    servoBrick2.set_position(5, denorm_actions[7])
    servoBrick2.set_enable(5, True)
    time.sleep(1.5)  # Modify to a value that is sufficient to go to target position in real pib
