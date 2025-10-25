import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import io

# Try to import terminal image display libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def display_image_in_terminal(fig=None, img_path=None, width=80):
    """Display an image in the terminal using ASCII art"""
    if not PIL_AVAILABLE:
        print("PIL not available for terminal display")
        return False
    
    try:
        # If figure is provided, save it to a temporary buffer
        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
        # If image path is provided, load it
        elif img_path is not None:
            img = Image.open(img_path)
        else:
            return False
        
        # Convert to grayscale for better ASCII representation
        img = img.convert('L')
        
        # Resize to fit terminal width
        height = int(width * img.height / img.width * 0.5)  # 0.5 to account for character height
        img = img.resize((width, height))
        
        # ASCII characters from dark to light
        ascii_chars = '@%#*+=-:. '
        
        # Convert image to ASCII
        pixels = np.array(img)
        ascii_str = ''
        for y in range(height):
            for x in range(width):
                pixel_value = pixels[y, x]
                ascii_str += ascii_chars[int(pixel_value / 255 * (len(ascii_chars) - 1))]
            ascii_str += '\n'
        
        print(ascii_str)
        return True
    except Exception as e:
        print(f"Error displaying image in terminal: {e}")
        return False

def plot_env(env, show_in_terminal=False, save_path=None):
    """Plot the basic environment"""
    grid = np.zeros((env.size, env.size))
    x, y = env.state
    gx, gy = env.goal
    
    # Handle obstacles if they exist
    if hasattr(env, 'obstacle'):
        for ox, oy in env.obstacle: 
            grid[ox, oy] = 1.0
    
    grid[gx, gy] = 0.4  # Mark goal
    grid[x, y] = 0.2    # Mark agent
    
    if show_in_terminal:
        # Simple ASCII representation
        print("\nEnvironment:")
        for i in range(env.size):
            row = ""
            for j in range(env.size):
                if (i, j) == (x, y):
                    row += "A "
                elif (i, j) == (gx, gy):
                    row += "G "
                elif hasattr(env, 'obstacle') and (i, j) in env.obstacle:
                    row += "O "
                else:
                    row += ". "
            print(row)
        print("A: Agent, G: Goal, O: Obstacle, .: Empty")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    
    # Add labels
    if hasattr(env, 'obstacle'):
        for ox, oy in env.obstacle: 
            ax.text(oy, ox, 'O', ha='center', va='center', color='white', fontsize=16)
    
    ax.text(gy, gx, 'G', ha='center', va='center', color='red', fontsize=16)
    ax.text(y, x, 'A', ha='center', va='center', color='black', fontsize=16)
    ax.set_title("GridWorld: Agent (A), Goal (G), Obstacles (O)")
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Environment visualization saved to {save_path}")
    
    # Try to display in terminal
    if not display_image_in_terminal(fig=fig):
        # If terminal display fails, show with GUI
        plt.show()
    
    plt.close(fig)

def visualize_policy(env, policy, V, show_in_terminal=False, save_path=None):
    """Visualize the optimal policy as arrows on a grid and superimpose V values"""
    if show_in_terminal:
        # Simple ASCII representation
        print("\nPolicy and Values:")
        for i in range(env.size):
            row = ""
            for j in range(env.size):
                state = (i, j)
                if hasattr(env, 'obstacle') and state in env.obstacle:
                    row += " O  "
                elif state == env.goal:
                    row += " G  "
                else:
                    action = policy.get(state, '?')
                    value = V.get(state, 0)
                    row += f"{action}/{value:.1f} "
            print(row)
        print("Action: U=Up, D=Down, L=Left, R=Right, G=Goal, O=Obstacle")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a grid for visualization
    grid = np.zeros((env.size, env.size))
    
    # Mark goal
    gx, gy = env.goal
    grid[gx, gy] = 0.5
    
    # Mark obstacles if they exist
    if hasattr(env, 'obstacle'):
        for ox, oy in env.obstacle:
            grid[ox, oy] = 0.8
    
    # Display the grid
    ax.imshow(grid, cmap='Blues', vmin=0, vmax=1)
    
    # Define arrow directions for each action
    arrows = {
        'U': (0, -0.3),
        'D': (0, 0.3),
        'L': (-0.3, 0),
        'R': (0.3, 0)
    }
    
    # Draw arrows and V values for each state
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            
            # Skip obstacles
            if hasattr(env, 'obstacle') and state in env.obstacle:
                continue
            
            # Display V value
            if state != env.goal:
                value = V.get(state, 0.0)
                ax.text(j, i - 0.2, f'{value:.2f}',
                        ha='center', va='center', fontsize=10, color='blue', weight='bold')
            
            # Draw policy arrows
            if state != env.goal:
                action = policy.get(state, None)
                if action in arrows:
                    dx, dy = arrows[action]
                    ax.arrow(j, i + 0.1, dx, dy, head_width=0.1, head_length=0.1, 
                            fc='red', ec='red', linewidth=2)
            else:
                ax.text(j, i, 'GOAL', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='green')
    
    # Add grid lines and labels
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Optimal Policy and Value Function Visualization', fontsize=14)
    ax.set_xlabel('Y coordinate')
    ax.set_ylabel('X coordinate')
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
    
    # Try to display in terminal
    if not display_image_in_terminal(fig=fig):
        # If terminal display fails, show with GUI
        plt.show()
    
    plt.close(fig)

def print_policy(env, policy):
    """Print the policy in a readable format"""
    print("Optimal Policy:")
    for state in env.get_all_states():
        if state != env.goal:
            print(f"  {state}: {policy[state]}")

def print_values(V):
    """Print the value function"""
    print("\nValue Function:")
    for state, value in sorted(V.items()):
        print(f"  {state}: {value:.2f}")


