# Controller Demo Scripts

This directory contains demonstration scripts for testing the controller → HAL → IsaacSimMCUSDK integration pipeline.

## Scripts


### 1. `test_gamepad_to_isaacsim_hal.py`

**Purpose**: Full end-to-end test that verifies the complete pipeline from gamepad input through the ControlLoop, HAL client/server, to IsaacSimMCUSDK logging.

**What it tests**:
- InputController reading gamepad events. 
- ControlLoop wiring and callback system
- GamepadToIsaacSimHALMapper conversion
- HALClient → HALServer communication
- IsaacSimMCUSDK logging in Isaac's preferred format

**Requirements**:
- A gamepad/joystick connected (Bluetooth or USB)
- The `pygame` library: `pip install pygame`

**Usage**:
```bash
# From the krabby-research root directory
python controller/scripts/demo/test_gamepad_to_isaacsim_hal.py
```

**Expected Output**: 
When you press buttons on the gamepad, you should see 
 **IsaacSimMCUSDK debug logs** showing joint commands in Isaac's format. When the test is run, it keeps on logging the joint commands in Isaac's format continuously. 
To see specific action, you can press a specific gamepad button(like LB, etc) and then move the joystick(like left stick, right stick, etc) to see the corresponding joint commands in Isaac's format.
   ```
   2026-01-15 11:50:07,234 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Applying joint command (timestamp_ns=1768495807227837000, observation_timestamp_ns=1768495807227837000): FL_hip_yaw=0.0000, FL_hip_pitch=0.0000, FL_knee=0.0000, FR_hip_yaw=0.0000, FR_hip_pitch=0.0000, FR_knee=0.0000, ML_hip_yaw=0.0000, ML_hip_pitch=0.0000, ML_knee=0.0000, MR_hip_yaw=0.0000, MR_hip_pitch=0.0000, MR_knee=0.0000, RL_hip_yaw=0.0000, RL_hip_pitch=0.0000, RL_knee=0.0000, RR_hip_yaw=0.0000, RR_hip_pitch=0.0000, RR_knee=0.0000
  ...
  -- This is after pressing a button and moving the sticks
  2026-01-15 11:50:51,912 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Applying joint command (timestamp_ns=1768495851908335000, observation_timestamp_ns=1768495851908335000): FL_hip_yaw=0.0000, FL_hip_pitch=0.0000, FL_knee=0.0000, FR_hip_yaw=0.0000, FR_hip_pitch=0.0000, FR_knee=0.0000, ML_hip_yaw=-0.0160, ML_hip_pitch=-0.5408, ML_knee=-2.0000, MR_hip_yaw=0.0000, MR_hip_pitch=0.0000, MR_knee=0.0000, RL_hip_yaw=-0.1159, RL_hip_pitch=1.7897, RL_knee=0.0174, RR_hip_yaw=0.0000, RR_hip_pitch=0.0000, RR_knee=0.0000
  2026-01-15 11:50:51,912 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Joint command stats - min=-2.0000, max=1.7897, mean=-0.0481, std=0.6441
   ```


**TODOs: 1/18/2026 - flliver@ feedback**

Overall:
Most of the way there conceptually, and 90% of the code is there now. Most critical issue is poor mapper abstraction (see below). Overall, I see the same common problems I always see w/ AI development these days:
1) Review code for ifdefs and conditionals, especially during instantiation, and remove all the crap code. AIs don't really understand the conceptual model and instatiation flow of what they are writing, so they tend to be extremely defensive to the point of being stupid. Just delete all that crap code and instantiate your class instance variables and use them normally, no need to check if they are present every single time if you're controlling the class constructor.
2) Anytime AI is writing a MagicMock, it's probably doing something wrong. i.e. you can just delete that entire test and only use the one that does the isaacMCUSDK test, that's the real test.
3) AI gets lost when passing through Object encapsulation, and doesn't really understand encapsulation as a concept at all. Check your inputcontroller, mapper, etc. and make sure it's only doing what it's actually supposed to do. i.e. your mapper right now is trying to apply business logic about joint commands instead of just mappnig from input -> isaac, your input controller is trying to clip the analog triggers instead of just passing values through to the mapper, etc. 
4) Some goofy AI driven choices like trying to clone the whole file to support pygame, instead of the obvious, which is just use pygame in the first place, or trying to do a 5 line import instead of just a pip install -e .. In general, review all the AI code for stuff that looks unnecessarily complex and delete, delete, delete until the code looks simple/obvious. Frequently you can just double check with the AI if this is really how it's supposd to be done as it looks too complex.


input_controller: Updated. &check;
* 171 - Use input validation and put 0.02 as default rather than ifdef w/ buried default value : Updated, no default needed as update_rate_hz value is validated 
* 218 - Rethrow rather than fail/crash silently : Updated. rethrow added
* 238 - Any reason not to also throw the triggers through the HAL as analog, and let the HAL mapper decide if they want a digital signal? Also, generally minimize the amount of input mapping happening here, that's what the input->isaac mapper is for. : Updated. all this is moved to the mapper 
* 253 - I think you're supposed to normalize to the gamepad's published absolute values rather than use hardcoded default (evdev.AbsInfo maybe?). : Updated. moved to the mapper
* 266 - **CRITICAL** This is conceptually incorrect encapsulation, the input controller should literally just be processing/assembling the inputs and passing it to the mapper. The mapper should be running all this input -> robot-limb-state embodiment mapping code. : Updated. this is updated


gamepad_to_isaacsim_hal_mapper.py: : Updated. &check;
* 24 - I usually don't like using array indices to specify something that has a semantic definition, i.e. I'd normally expect this to be a nested map where 'FRONT_LEFT' is contains its own little map of HY, HL, KL, or at least a flat map of FLHY, FLHL, FLKL (front-left hip yaw, ...). That said, this is simple/easy to understand so it's fine. In general though if something has a semantic meaning, try to represent that semantic meaning directly with the data structure used, i.e. if you mean 'Front left Hip Yaw', you should do something like 'joints['FLHY']', or 'joints['FL']['HY']', not 'joints[0]', that way there's no ambiguity or possibility to grab the wrong joint.
* 104: **CRITICAL** This impl doesn't look conceptually correct. The map() function should do exactly that, map from input controller to whatever isaac expects, then the HAL server impl should be applying the mapped inputs to isaac (in isaacsim_mcusdk.py). It should not be applying any controls, and definitely should not be having global values that caculate leg movement based on deltas or anything like that. Delete all this and let isaacsim_mcusdk.py decide how to apply the control commands.  : Updated. removed deltas
* 136 - **CRITICAL** I don't think that's right, I think all linear joints will be 0.0 - 1.0 range, and I don't think you need to clamp at all. Generally this code should not be doing any business logic, it should just map from the input controller values to the values needed by isaac sim. : Updated. removed clamping

control_loop.py: : Updated. &check;
* 16 - Could we use a conditional import and keep mac support if that's the only place that needs to change? (huzzah for interpreted languages!) : Updated. not needed any more as only Pygame is used
* 168:186 - This looks like AI written code trying to be overly cautious and add all this weird ifdef handling to try to hedge against different initialization configurations that would never actually happen. Delete all this and just initialize w/ no ifdefs. Also, not clear why HAL is leaking the ZMQContext at all, see if you can just add a constructor that handles the ZMQContext internally. : Updated. updated it
* 224: Doesn't seem like this should be a warning? I'd just delete this line and let it throw NPE as this should never be reachable (AI makes this mistake commonly, trying to validate inputs and swallow errors, because it doesn't actualy understand what the overall class init state should be) : this is updated to not do the check and throw NPE
* 230: Add official TODO in comment so it's trackable : Updated. added a TODO

input_controller_test_pygame.py: Updated. &check; This is removed now.
* 1: Just use pygame if it supports both linux and mac, then delete this whole file and we can develop on mac or linux (note, I've been developing on windows, so if windows works too, then we're really winning and making ourselves robust/platform agnostic, which is a good thing) : Updated. only input_controller for Pygame used now. Not using inputs library

state.py: Updated. &check;
* 53-55: **CRITICAL** I don't really like these names, they are already starting to map functionality to the specific robot embodiment. Imagine reusing this controller class for a different robot that had no knees, this would need to be refactored. InputController should literally just be storing the state of the gamepad in a @dataclass to give to the mapper, which applys the joint mapping based on the A) Robot embodiment, B) input control type, C) output environment type. Complete encapsulation inside the mapper of the input+embodiment+environment. This will let you delete/clean alot of code out of input_controller.py : Updated. with the mapper reefactor, all this is updated


test_gamepad_to_isaacsim_hal_mapper.py
* 85-113: LOL, beast of a mock. I have a general ban on AI usage of MagicMock(), too many times the AI just mocks out everything and you actually test nothing at all. You shouldn't need any of this either, all you should do is check that IsaacSimMCUSDK gets properly instantiated and called w/ the correct mapped values. That way your test is directly testing what's supposed to be tested, the integration of the HAL into the IsaacSimMCUSDK.

test_mapper_sdk_logging.py : Updated. &check; This is removed. There is a unit test testing similar functionality.
* 19-25: This looks like a weird AI hack, I think you're supposed to pip install -e . the project dir or run it as a module w/ python -m controller.scripts....
* 70-71: I don't really like up/down in/out, been using 

isaacsim_mcusdk.py: Updated. &check;   
** 23: Isaac doesn't take a torch tensor, that's AI model stuff. Isaac's core representation of these joints will be as prismatic joints that have a position/velocity target w/ some form of stiffness/damping/force effects configured on the joint. In a later task, someone will build the IsaacSim URDF of the robot and configure those physics elements for the joint. At this point all you need to do is hand over the -1.0:1.0 'normalized PWM' value coming from the input controller (i.e. a 0.0 means keep the joint in current position, 1.0 means move the joint out as fast as possible, -0.5 means move the joint in at 50% speed). This is conveniently pretty much a no-op because you've already setup the input controller to give a normalized analog value for the stick axis (i.e. the stick all the way left is -1.0 is the same as 'move joint all the way in'): Updated
** 43-119: Doesn't actually do anything w/ isaac right now, so needs to have a 'TODO: Add isaac code to control prismatic joints here': Updated
** 43-119: More torch references that don't belong, go directly from mapped values into the prismatic joints: Updated. Torch references removes. hal/server/isaac/hal_server.py updated to convert the return value of mcusdk.apply_command to Torch Tensor as this is what the calling code is expecting.
** 121: I dunno what this set_device is for, I'd remove it for now if it's not serving a purpose, too much superfluous code has this compounding effect of confusing AI in later iterations. : updated. This is removed. 
 

Tests: 
* Didn't read in detail, but has normal AI problems, where it's just patching everything with mocks and testing nothing, and has all these 'edge cases' that aren't really edge cases once the code is simplified and has proper object/class abstraction. Once you clean up and keep the main business logic inside the mapper, I think you'll delete more than half of them, so don't let the AI spazz out and make tests more complex than needed.

test_input_controller.py: Updated. &check; All these are updated along with the mapper refactor. 
* 529: I'm not sure this actually makes things that concurrent becuase it uses fixed sleep intervals. Not a terrible test though.
* 801: This is the kind of stuff you'll be able to delete once you've got the right encapsulation w/ the mapper.

test_gamepad_to_isaacsim_hal_mapper.py:Updated. &check; All these are updated along with the mapper refactor. 
* Update all these after refactor, main thing is to ensure that input controller button/stick values get turned into 18 joint directional commands from -1.0 -> 1.0. Should be really easy though to write one test that presses every possible combination of button/direction and ensures output comes out the back based on the button(s) pressed. That would shorten this to maybe 100 lines and catch all edge cases w/o special handling. Only specific special tests required would be the double/quad shoulder triggers. 

test_control_loop.py: Updated. &check; All these are updated along with the mapper refactor. 
* I dunno what I think about this extensive patching, it's not really testing anything anymore. I'd just test the actual initialization w/ real classes and drop all this mock nonsense.
