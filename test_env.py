import src.env
import gymnasium as gym

def main(env_family: str, env_name: str):
    
    path_txt_grid = f'./src/env/grid/txts/{env_name}.txt'

    env = gym.make(
        env_family, 
        path=path_txt_grid, 
        render_mode="human", 
        use_target=True, 
    )

    print(env.grid.shape)
    observation, info = env.reset()

    for _ in range(100000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    env_family = 'Grid-v0'
    env_name  = 'GridRoom-64'

    main(env_family, env_name)