import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_offers.envs.offer_env import OfferEnv

def make_env():
    # TODO: load real data
    num_customers = 10000
    feature_dim = 20
    customer_df = np.random.randn(num_customers, feature_dim).astype(np.float32)

    offer_catalog = [{"id": i, "value": i*0.5} for i in range(1, 6)]  # 5 offers
    channel_costs = {0:0.0, 1:0.02, 2:0.03, 3:0.05, 4:0.07}

    env = OfferEnv(customer_df=customer_df,
                   offer_catalog=offer_catalog,
                   channel_costs=channel_costs,
                   weekly_budget=1e6,
                   horizon_weeks=12,
                   lambda_trips=0.5)
    return env

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb")
    model.learn(total_timesteps=200_000)
    model.save("ppo_offer_policy.zip")
