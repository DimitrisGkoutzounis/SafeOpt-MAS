from gym.envs.registration import register

register(
    id='Env-v0',
    entry_point='pandaenv.envs:MyEnv',  
    order_enforce=False, 
)

