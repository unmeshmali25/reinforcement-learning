# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # DATA AND DATA ANALYSIS

# %%
# Load the datasets (adjust the path to where you download the files)
print("Loading MovieLens 100k data...")
# Download from: https://files.grouplens.org/datasets/movielens/ml-100k.zip
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')


# %%
users.head()

# %%
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

# %%
ratings.head()

# %%
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('./data/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')

# %%
movies.head()

# %%
# Merge the data into one comprehensive DataFrame
print("Merging data...")
df = ratings.merge(users, on='user_id').merge(movies, on='movie_id')

# Basic Dataset Info
print("\n=== DATASET SHAPE ===")
print(f"Ratings DataFrame Shape: {df.shape}")
print(f"Unique Users: {df['user_id'].nunique()}")
print(f"Unique Movies: {df['movie_id'].nunique()}")

# %%
# Basic Statistics
print("\n=== BASIC STATISTICS ===")
print(df['rating'].describe())

# %%
# Visualize the distribution of ratings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df.groupby('user_id')['rating'].count().plot(kind='hist', bins=30, color='salmon')
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.savefig('rating_distribution.png')
print("Plot saved as 'rating_distribution.png'")

# %%
# Explore genres (the core "features" of our movies)
genre_columns = movies.columns[5:] # Get all genre columns
movie_genre_counts = movies[genre_columns].sum().sort_values(ascending=False)

# %%
movie_genre_counts.head()

# %%
plt.figure(figsize=(12, 6))
movie_genre_counts.plot(kind='bar', color='lightgreen')
plt.title('Number of Movies per Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('genre_distribution.png')
print("Plot saved as 'genre_distribution.png'")

# %%
print("\n=== TOP 10 MOST RATED MOVIES ===")
top_movies = df['title'].value_counts().head(10)
print(top_movies)

# %% [markdown]
# # Preparing data for bandit

# %%
# Merge to get movie titles
df = ratings.merge(movies, on='movie_id')

# %%
# Define a binary reward: 1 if rating >= 4, else 0
df['reward'] = (df['rating'] >= 4).astype(int)

# %%
# ----------------------------
# 2. DEFINE OUR ARMS (ACTION SPACE)
# ----------------------------
# Let's choose the top 50 most frequently rated movies to start with.
# This creates a manageable action space for our first bandit.
top_movies = df['movie_id'].value_counts().head(50).index.tolist()
# Filter the dataset to only include interactions with these top movies
df_top = df[df['movie_id'].isin(top_movies)].copy()

# Create a mapping from movie_id to a simpler arm index (0 to 49)
arm_index_map = {movie_id: idx for idx, movie_id in enumerate(top_movies)}
df_top['arm'] = df_top['movie_id'].map(arm_index_map)

n_arms = len(top_movies)
print(f"\nWorking with a subset of {n_arms} arms (movies).")
print(f"Number of interactions in this subset: {len(df_top)}")

# %%
# ----------------------------
# 3. THE THOMPSON SAMPLING AGENT
# ----------------------------
class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        # We start with a Beta(1, 1) prior for each arm, which is uniform
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.n_arms = n_arms
        
    def select_arm(self):
        # Sample a value from the Beta distribution for each arm
        theta_samples = np.random.beta(self.alpha, self.beta)
        # Return the arm with the highest sampled value
        return np.argmax(theta_samples)
    
    def update(self, chosen_arm, reward):
        # Update the Beta distribution parameters for the chosen arm
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

# %%
# ----------------------------
# 4. SIMULATE THE ONLINE LEARNING LOOP
# ----------------------------
# We will run through the historical data in chronological order.
# For each recorded interaction, our bandit will choose an arm (movie).
# We then update the bandit with the reward from the *actually* watched movie.
# This simulates what would happen if we had deployed our bandit.

print("\nSimulating the online learning loop...")
df_top_sorted = df_top.sort_values('unix_timestamp') # Simulate time

# Initialize the bandit and track performance
bandit = ThompsonSamplingBandit(n_arms)
random_cumulative_rewards = []
bandit_cumulative_rewards = []
cumulative_reward = 0
bandit_cumulative_reward = 0

# We'll run the simulation on a subset for speed (optional)
# df_top_sorted = df_top_sorted.head(10000)

# %%
for i, row in df_top_sorted.iterrows():
    user_id, movie_id, true_arm, true_reward = row['user_id'], row['movie_id'], row['arm'], row['reward']
    
    # 1. Bandit chooses an arm
    chosen_arm = bandit.select_arm()
    
    # 2. We observe the reward for the arm that was *actually* taken in the historical data.
    # In a true online setting, we would get a reward for the chosen_arm.
    # For this simulation, we use the "true_reward" only if the bandit happened to choose
    # the same movie the user actually watched. This is a common way to evaluate on historical data.
    if chosen_arm == true_arm:
        reward_for_update = true_reward
        bandit_cumulative_reward += true_reward
    else:
        # If the bandit chose a different movie, we don't get to observe the reward for its choice.
        # We only know what happened for the movie the user actually watched.
        reward_for_update = None # We can't update for an outcome we didn't observe
    
    # 3. Update the bandit with the outcome, but only if we have a reward for the chosen action
    if reward_for_update is not None:
        bandit.update(chosen_arm, reward_for_update)
    
    # 4. Track cumulative reward for the bandit and a random policy
    cumulative_reward += true_reward # This is the reward from the historical policy
    random_cumulative_rewards.append(cumulative_reward / (i+1)) # Running average for random
    bandit_cumulative_rewards.append(bandit_cumulative_reward / (i+1)) # Running average for bandit

# ----------------------------
# 5. PLOT AND ANALYZE RESULTS
# ----------------------------
print("\nPlotting results...")
plt.figure(figsize=(12, 6))
plt.plot(bandit_cumulative_rewards, label='Thompson Sampling', linewidth=2)
plt.plot(random_cumulative_rewards, label='Historical (Random) Policy', linestyle='--', alpha=0.8)
plt.xlabel('Number of Rounds (User Interactions)')
plt.ylabel('Cumulative Average Reward')
plt.title('Thompson Sampling vs. Random Policy Performance (Simulated on Historical Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('thompson_sampling_performance.png', dpi=150)
plt.show()

# Analyze final performance
final_ts_reward = bandit_cumulative_rewards[-1]
final_random_reward = random_cumulative_rewards[-1]
improvement = ((final_ts_reward - final_random_reward) / final_random_reward) * 100

print("\n=== SIMULATION RESULTS ===")
print(f"Final Cumulative Average Reward (Thompson Sampling): {final_ts_reward:.4f}")
print(f"Final Cumulative Average Reward (Historical Policy): {final_random_reward:.4f}")
print(f"Improvement: {improvement:+.2f}%")

# Let's see which movies the bandit learned were the best
print("\n=== TOP 5 RECOMMENDED MOVIES (Highest Estimated Click-Through Rate) ===")
# The estimated probability of success (reward=1) is alpha / (alpha + beta)
arm_probs = bandit.alpha / (bandit.alpha + bandit.beta)
top_5_arms = np.argsort(arm_probs)[-5:][::-1] # Indices of top 5 arms

# Map arm indices back to movie titles
for arm_idx in top_5_arms:
    original_movie_id = top_movies[arm_idx]
    movie_title = movies.loc[movies['movie_id'] == original_movie_id, 'title'].iloc[0]
    est_prob = arm_probs[arm_idx]
    print(f"Movie: {movie_title: <50} | Estimated P(Like): {est_prob:.3f}")

print("\nSimulation complete!")

# %%



