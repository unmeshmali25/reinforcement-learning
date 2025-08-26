# rl_offers/envs/offer_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OfferEnv(gym.Env):
    """
    Weekly offer selection environment.
    You call reset() at the start of a 'season' (H weeks).
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 customer_df,
                 offer_catalog,
                 channel_costs,
                 weekly_budget,
                 horizon_weeks=12,
                 lambda_trips=0.5,
                 seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.rng = rng

        # --- Save config
        self.customer_df = customer_df              # preprocessed feature matrix (N_customers x F)
        self.offer_catalog = offer_catalog          # list/dict of offers with cost info
        self.channel_costs = channel_costs          # dict: channel_id -> cost
        self.weekly_budget_init = weekly_budget
        self.horizon_weeks = horizon_weeks
        self.lambda_trips = lambda_trips

        self.num_customers = customer_df.shape[0]
        self.num_offers = len(offer_catalog) + 1     # +1 for "no offer"
        self.num_channels = 5

        # --- Observation space
        # Example: concat customer features + [week_idx, budget_left_frac]
        self.num_features = customer_df.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)

        # --- Action space:
        # pick ONE offer (0 = no offer, 1..K = offer id) and ONE channel (0..4)
        self.action_space = spaces.MultiDiscrete([self.num_offers, self.num_channels])

        # Internal state
        self.week_idx = 0
        self.budget_left = self.weekly_budget_init
        self.curr_customer_idx = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.week_idx = 0
        self.budget_left = self.weekly_budget_init
        # In this simple version, iterate customers; one step = 1 customer-week
        self.curr_customer_idx = 0

        obs = self._make_obs(self.curr_customer_idx)
        return obs, {}

    def step(self, action):
        offer_id, channel_id = action

        # --- Get state pieces
        cust_idx = self.curr_customer_idx
        week = self.week_idx

        # --- Simulate outcome (replace with your causal/propensity models)
        outcome = self._simulate_outcome(cust_idx, offer_id, channel_id)

        profit = outcome["profit"]
        trips_delta = outcome["trips_delta"]
        discount_cost = outcome["discount_cost"]
        channel_cost = self.channel_costs[channel_id]

        # Budget check
        spend = discount_cost + channel_cost
        if spend > self.budget_left:
            # Penalty for exceeding budget
            reward = -5.0
        else:
            self.budget_left -= spend
            reward = profit + self.lambda_trips * trips_delta

        # Advance pointer
        done = False
        info = {"profit": profit,
                "trips_delta": trips_delta,
                "budget_left": self.budget_left,
                "spend": spend}

        self.curr_customer_idx += 1
        if self.curr_customer_idx >= self.num_customers:
            # end of week
            self.curr_customer_idx = 0
            self.week_idx += 1
            # reset weekly budget
            self.budget_left = self.weekly_budget_init

        if self.week_idx >= self.horizon_weeks:
            done = True

        obs = self._make_obs(self.curr_customer_idx)

        return obs, reward, done, False, info

    # ----------------- Helpers -----------------

    def _make_obs(self, cust_idx):
        cust_feats = self.customer_df[cust_idx]  # np.array
        obs = np.concatenate([
            cust_feats,
            np.array([self.week_idx / self.horizon_weeks,
                      self.budget_left / self.weekly_budget_init], dtype=np.float32)
        ])
        return obs.astype(np.float32)

    def _simulate_outcome(self, cust_idx, offer_id, channel_id):
        """
        Stub: plug your trained predictive models here.
        Should return dict with profit, trips_delta, discount_cost.
        """
        # TODO: call calibrated models: P(trip|context,offer), expected spend, margin, etc.
        # For now, random toy logic:
        base_trips = 0.05
        base_profit = 0.5
        offer_boost = 0.02 * offer_id        # increasing effect for demo
        channel_boost = 0.005 * channel_id

        trips_delta = self.rng.binomial(1, base_trips + offer_boost + channel_boost)
        profit = base_profit + 0.3 * offer_id - 0.1 * channel_id

        discount_cost = 0.5 * offer_id  # dummy

        return {"profit": profit, "trips_delta": trips_delta, "discount_cost": discount_cost}
