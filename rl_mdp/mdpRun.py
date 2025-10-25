import argparse
from mdpGridworldEnvs import create_environment
from mdpPolicies import create_policy
from mdpVisualize import plot_env, visualize_policy, print_policy, print_values


def main():
    parser = argparse.ArgumentParser(description='Run MDP algorithms on GridWorld environments')
    parser.add_argument('--env', type=str, default='basic', 
                       choices=['basic', 'obstacle', 'stochastic'],
                       help='Type of environment')
    parser.add_argument('--policy', type=str, default='value_iteration',
                       choices=['value_iteration', 'policy_iteration'],
                       help='Policy algorithm to use')
    parser.add_argument('--size', type=int, default=4,
                       help='Size of the grid (size x size)')
    parser.add_argument('--obstacle', type=str, nargs='+', default=None,
                       help='Obstacle positions as "x,y" pairs')
    parser.add_argument('--gamma', type=float, default=0.9,
                       help='Discount factor')
    parser.add_argument('--theta', type=float, default=1e-6,
                       help='Convergence threshold')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualizations')
    parser.add_argument('--terminal', action='store_true',
                       help='Show visualizations in terminal')
    parser.add_argument('--print', action='store_true',
                       help='Print policy and values')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualizations to file (prefix)')
    
    args = parser.parse_args()
    
    # Parse obstacle positions
    obstacle = None
    if args.obstacle:
        obstacle = [tuple(map(int, pos.split(','))) for pos in args.obstacle]
    
    # Create environment
    print(f"Creating {args.env} environment of size {args.size}x{args.size}")
    env = create_environment(args.env, size=args.size, obstacle=obstacle)
    
    # Show environment
    if args.visualize or args.terminal:
        env_save_path = f"{args.save}_env.png" if args.save else None
        plot_env(env, show_in_terminal=args.terminal, save_path=env_save_path)
    
    # Run policy algorithm
    print(f"Running {args.policy} algorithm...")
    V, policy = create_policy(args.policy, env, gamma=args.gamma, theta=args.theta)
    
    # Print results
    if args.print:
        print_policy(env, policy)
        print_values(V)
    
    # Visualize policy
    if args.visualize or args.terminal:
        policy_save_path = f"{args.save}_policy.png" if args.save else None
        visualize_policy(env, policy, V, show_in_terminal=args.terminal, save_path=policy_save_path)


if __name__ == "__main__":
    main()



