import minatar
import gymnasium as gym
import numpy as np

class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = gym.make('MinAtar/'+env_name)#minatar.Environment(env_name) 
        #self.minimal_actions = minimal_action_set()
        #print("scree", self.env.observation_space.shape)
        h,w,c = self.env.observation_space.shape#state_shape()
        self.action_space = self.env.action_space#gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c,h,w))
        #print(self.observation_space.shape)

    def reset(self):
        return self.env.reset()
        #return self.env.state().transpose(2, 0, 1)
    
    def step(self, action):
        '''index is the action id, considering only the set of minimal actions'''
        #action = index#self.minimal_actions[index]
        #r, terminal = self.env.act(action)
        obs, r, terminal, truncated, done = self.env.step(action)
        self.game_over = terminal
        return obs, r, terminal, truncated, done

    def seed(self, seed='None'):
        self.env = gym.make('MinAtar/'+self.env_name, seed=seed)#minatar.Environment(self.env_name, random_seed=seed)
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0

class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))

    def observation(self, observation):
        return np.stack([observation[:,:,0], observation[:,:,1], observation[:,:,3]], axis=0)
    
class asterixPOMDP(gym.ObservationWrapper):
    '''index 2 (trail) is removed, which gives ball's direction'''
    def __init__(self, env):
        super(asterixPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class freewayPOMDP(gym.ObservationWrapper):
    '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''
    def __init__(self, env):
        super(freewayPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-5,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1]], axis=0)    

class space_invadersPOMDP(gym.ObservationWrapper):
    '''index 2-3 (trail) are removed, which gives aliens' direction'''
    def __init__(self, env):
        super(space_invadersPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-2,h,w))
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)

class seaquestPOMDP(gym.ObservationWrapper):
    '''index 3 (trail) is removed, which gives enemy and driver's direction'''
    def __init__(self, env):
        super(seaquestPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
        
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6], observation[7], observation[8], observation[9]], axis=0)    

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self, *args, **kwargs):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
