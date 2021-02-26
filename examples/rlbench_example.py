from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, CloseBox, StackBlocks
from enum import Enum
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def ingest(self, demos):
        pass

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, DATASET, obs_config, headless=False,
    robot_configuration='ur3')
env.launch()

class Algos(Enum):
    BiTRRT = 'BiTRRT'
    BITstar = 'BITstar'
    BKPIECE1 = 'BKPIECE1'
    CForest = 'CForest'
    EST = 'EST'
    FMT = 'FMT'
    KPIECE1 = 'KPIECE1'
    LazyPRM = 'LazyPRM'
    LazyPRMstar = 'LazyPRMstar'
    LazyRRT = 'LazyRRT'
    LBKPIECE1 = 'LBKPIECE1'
    LBTRRT = 'LBTRRT'
    PDST = 'PDST'
    PRM = 'PRM'
    PRMstar = 'PRMstar'
    pRRT = 'pRRT'
    pSBL = 'pSBL'
    RRT = 'RRT'
    RRTConnect = 'RRTConnect'
    RRTstar = 'RRTstar'
    SBL = 'SBL'
    SPARS = 'SPARS'
    SPARStwo = 'SPARStwo'
    STRIDE = 'STRIDE'
    TRRT = 'TRRT'

task = env.get_task(StackBlocks)
demos = task.get_demos(5, live_demos=live_demos, max_attempts=1)

agent = Agent(env.action_size)
agent.ingest(demos)

# training_steps = 120
# episode_length = 40
# obs = None
# for i in range(training_steps):
#     if i % episode_length == 0:
#         print('Reset Episode')
#         descriptions, obs = task.reset()
#         print(descriptions)
#     action = agent.act(obs)
#     print(action)
#     obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()