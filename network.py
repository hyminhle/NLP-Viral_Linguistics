"""
network_sir_model.py - Network-enhanced Linguistic Contagion Model

This module extends the basic SIR model with network dynamics to better model
how linguistic innovations spread differently on platforms like Twitter and Reddit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from sir_model import LinguisticContagionModel, SIRParameters


class NetworkLCM(LinguisticContagionModel):
    """
    Network-aware Linguistic Contagion Model that extends the basic SIR framework
    to incorporate different network structures across platforms.
    """
    
    def __init__(self, params=None, network_type="random"):
        """
        Initialize with network parameters
        
        Args:
            params: SIR parameters
            network_type: Type of network ("random", "scale_free", "community", etc.)
        """
        super().__init__(params)
        self.network_type = network_type
        self.network = None
        self.node_states = None  # Will store S, I, R states for each node
        
        # Platform-specific parameters
        self.platform_params = {
            "twitter": {
                "mean_followers": 707,  # Average followers per user
                "influencer_threshold": 10000,  # Threshold to be considered influencer
                "influencer_boost": 5.0,  # Multiplier for influencer transmission
                "hashtag_bridge_probability": 0.2,  # Probability of cross-community exposure
                "character_limit_effect": 1.2  # Effect of character limit on innovation
            },
            "reddit": {
                "subreddit_isolation": 0.7,  # How isolated communities are (0-1)
                "vote_threshold": 5,  # Upvotes needed for visibility
                "thread_depth_penalty": 0.15,  # Transmission reduction per depth level
                "moderator_density": 0.02,  # Proportion of users who are moderators
                "community_resistance": 0.5  # Community resistance to outside terms
            }
        }
    
    def generate_network(self, N, platform="twitter"):
        """
        Generate network structure based on platform characteristics
        
        Args:
            N: Number of nodes/users
            platform: "twitter" or "reddit"
        """
        if platform == "twitter":
            # Scale-free network to model follower structure (preferential attachment)
            self.network = nx.scale_free_graph(N)
            
            # Add influencer nodes
            influencers = []
            for node in self.network.nodes():
                if self.network.out_degree(node) > self.platform_params["twitter"]["influencer_threshold"]:
                    influencers.append(node)
                    
            # Add hashtag bridges (connections across communities)
            num_hashtag_bridges = int(N * self.platform_params["twitter"]["hashtag_bridge_probability"])
            for _ in range(num_hashtag_bridges):
                u = np.random.randint(0, N)
                v = np.random.randint(0, N)
                self.network.add_edge(u, v, type="hashtag")
                
        elif platform == "reddit":
            # Generate community structure with subreddits
            num_communities = int(np.sqrt(N) / 2)  # Heuristic for number of communities
            community_sizes = np.random.power(1.5, num_communities)
            community_sizes = (community_sizes / community_sizes.sum() * N).astype(int)
            
            # Ensure we have exactly N nodes
            while sum(community_sizes) < N:
                community_sizes[np.random.randint(0, len(community_sizes))] += 1
            while sum(community_sizes) > N:
                idx = np.random.randint(0, len(community_sizes))
                if community_sizes[idx] > 1:
                    community_sizes[idx] -= 1
            
            # Create communities with higher internal vs external connectivity
            self.network = nx.Graph()
            node_id = 0
            community_map = {}
            
            for i, size in enumerate(community_sizes):
                # Create subreddit community
                subreddit = nx.dense_gnm_random_graph(
                    size,
                    int(size * size * (1 - self.platform_params["reddit"]["subreddit_isolation"]))
                )
                
                # Map community nodes to global IDs
                mapping = {j: node_id + j for j in range(size)}
                subreddit = nx.relabel_nodes(subreddit, mapping)
                
                # Track community membership
                for node in subreddit.nodes():
                    community_map[node] = i
                
                # Add to main network
                self.network.add_nodes_from(subreddit.nodes())
                self.network.add_edges_from(subreddit.edges())
                
                # Add moderators
                num_mods = max(1, int(size * self.platform_params["reddit"]["moderator_density"]))
                mods = np.random.choice(list(range(node_id, node_id + size)), num_mods, replace=False)
                for mod in mods:
                    self.network.nodes[mod]['moderator'] = True
                
                node_id += size
            
            # Add cross-community connections (users in multiple subreddits)
            cross_connection_factor = 1 - self.platform_params["reddit"]["subreddit_isolation"]
            num_cross_edges = int(N * cross_connection_factor / 5)
            
            for _ in range(num_cross_edges):
                u = np.random.randint(0, N)
                # Prefer connecting to a different community
                v_community = (community_map[u] + np.random.randint(1, num_communities)) % num_communities
                v_nodes = [n for n, c in community_map.items() if c == v_community]
                if v_nodes:
                    v = np.random.choice(v_nodes)
                    self.network.add_edge(u, v, type="cross_community")
        
        else:
            # Default to random network
            self.network = nx.erdos_renyi_graph(N, 5/N)  # Average degree of 5
        
        # Initialize node states
        self.node_states = {node: "S" for node in self.network.nodes()}
        
        return self.network
    
    def simulate_network_spread(self, I0, N, platform="twitter", t_max=100, dt=1.0):
        """
        Simulate linguistic innovation spread through network
        
        Args:
            I0: Initial infected nodes
            N: Total population
            platform: Platform type
            t_max: Maximum time steps
            dt: Time increment
            
        Returns:
            DataFrame with time series of S, I, R counts
        """
        # Generate appropriate network if not already done
        if self.network is None or len(self.network.nodes()) != N:
            self.generate_network(N, platform)
        
        # Set initial infected nodes (preferentially target influential nodes for realism)
        initial_infected = []
        
        if platform == "twitter":
            # Prefer high-degree nodes (influencers) as initial adopters
            degrees = dict(self.network.degree())
            nodes_by_degree = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            initial_infected = nodes_by_degree[:I0]
        else:
            # Random selection across communities for Reddit
            initial_infected = np.random.choice(list(self.network.nodes()), I0, replace=False)
        
        for node in initial_infected:
            self.node_states[node] = "I"
        
        # Track statistics over time
        history = defaultdict(list)
        t_points = np.arange(0, t_max, dt)
        
        # Store initial state
        S_count = sum(1 for state in self.node_states.values() if state == "S")
        I_count = sum(1 for state in self.node_states.values() if state == "I")
        R_count = sum(1 for state in self.node_states.values() if state == "R")
        
        history['t'].append(0)
        history['S'].append(S_count)
        history['I'].append(I_count)
        history['R'].append(R_count)
        
        # Platform-specific parameter adjustments
        local_params = self.params.__dict__.copy()
        
        if platform == "twitter":
            # Twitter's network effects
            local_params['beta'] *= 1.2  # Higher transmission due to retweet mechanism
            local_params['gamma'] *= 1.5  # Higher re-susceptibility due to timeline algorithm
        elif platform == "reddit":
            # Reddit's network effects
            local_params['beta'] *= 0.8  # Lower transmission due to community boundaries
            local_params['nu'] *= 1.3    # Higher mutation due to community-specific adaptations
        
        # Simulation loop
        for t in t_points[1:]:
            # Copy current state to avoid simultaneous updates
            new_states = self.node_states.copy()
            
            # Process each node
            for node in self.network.nodes():
                if self.node_states[node] == "S":
                    # Susceptible nodes can become infected
                    
                    # Count infected neighbors
                    infected_neighbors = sum(
                        1 for neighbor in self.network.neighbors(node) 
                        if self.node_states[neighbor] == "I"
                    )
                    
                    # Platform-specific transmission dynamics
                    transmission_prob = local_params['beta'] * dt
                    
                    if platform == "twitter":
                        # Check if any infected neighbors are influencers
                        for neighbor in self.network.neighbors(node):
                            if (self.node_states[neighbor] == "I" and 
                                self.network.out_degree(neighbor) > 
                                self.platform_params["twitter"]["influencer_threshold"]):
                                # Influencer effect
                                transmission_prob *= self.platform_params["twitter"]["influencer_boost"]
                                break
                    
                    elif platform == "reddit":
                        # Check community boundaries
                        if 'community_map' in locals():
                            same_community_infected = sum(
                                1 for neighbor in self.network.neighbors(node)
                                if (self.node_states[neighbor] == "I" and 
                                    community_map[neighbor] == community_map[node])
                            )
                            # Different transmission rates within vs across communities
                            if infected_neighbors > same_community_infected:
                                transmission_prob *= (1 - self.platform_params["reddit"]["community_resistance"])
                            
                            # Moderator effect (if node is a moderator)
                            if self.network.nodes[node].get('moderator', False):
                                transmission_prob *= 0.7  # Moderators less likely to adopt random terms
                    
                    # Calculate infection probability
                    p_infection = 1 - (1 - transmission_prob) ** infected_neighbors
                    
                    # Attempt infection
                    if np.random.random() < p_infection:
                        new_states[node] = "I"
                    
                    # Random exposure to new users (μ term)
                    elif np.random.random() < local_params['mu'] * dt:
                        new_states[node] = "I"
                
                elif self.node_states[node] == "I":
                    # Infected nodes can recover or mutate
                    
                    # Recovery
                    if np.random.random() < local_params['alpha'] * dt:
                        new_states[node] = "R"
                    
                    # Mutation (term changes meaning/form)
                    elif np.random.random() < local_params['nu'] * dt:
                        # In our model, mutation still counts as recovery
                        new_states[node] = "R"
                
                elif self.node_states[node] == "R":
                    # Recovered nodes can become susceptible again
                    if np.random.random() < local_params['gamma'] * dt:
                        new_states[node] = "S"
            
            # Update all states
            self.node_states = new_states
            
            # Count statistics
            S_count = sum(1 for state in self.node_states.values() if state == "S")
            I_count = sum(1 for state in self.node_states.values() if state == "I")
            R_count = sum(1 for state in self.node_states.values() if state == "R")
            
            history['t'].append(t)
            history['S'].append(S_count)
            history['I'].append(I_count)
            history['R'].append(R_count)
        
        # Convert to DataFrame
        self.history = pd.DataFrame(history)
        self.N = N
        
        return self.history
    
    def visualize_network_state(self, t_index=None, figsize=(12, 10)):
        """
        Visualize the current state of the network
        
        Args:
            t_index: Time index to visualize (if None, uses current state)
            figsize: Figure size
        """
        if self.network is None:
            raise ValueError("Must generate network first")
        
        plt.figure(figsize=figsize)
        
        # Define colors for each state
        color_map = {"S": "blue", "I": "red", "R": "green"}
        
        # Get node colors
        node_colors = [color_map[self.node_states[node]] for node in self.network.nodes()]
        
        # Position nodes
        pos = nx.spring_layout(self.network)
        
        # Draw network
        nx.draw_networkx_nodes(self.network, pos, node_size=50, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(self.network, pos, alpha=0.1)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          label=label, markerfacecolor=color, markersize=10)
                          for label, color in color_map.items()]
        plt.legend(handles=legend_elements, title="Node States")
        
        # Add statistics
        S_count = sum(1 for state in self.node_states.values() if state == "S")
        I_count = sum(1 for state in self.node_states.values() if state == "I")
        R_count = sum(1 for state in self.node_states.values() if state == "R")
        
        stats_text = f"S: {S_count} ({S_count/self.N:.1%})\n"
        stats_text += f"I: {I_count} ({I_count/self.N:.1%})\n"
        stats_text += f"R: {R_count} ({R_count/self.N:.1%})"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.title("Network State of Linguistic Innovation Spread", fontsize=16)
        plt.axis('off')
        
        return plt.gcf()
    
    def compare_platforms(self, I0, N, platforms=["twitter", "reddit"], t_max=100):
        """
        Compare linguistic spread across different platforms
        
        Args:
            I0: Initial infected count
            N: Population size
            platforms: List of platforms to compare
            t_max: Maximum simulation time
            
        Returns:
            Comparison plot and summary DataFrame
        """
        results = {}
        histories = {}
        
        for platform in platforms:
            # Reset model
            self.network = None
            self.node_states = None
            
            # Run simulation for this platform
            history = self.simulate_network_spread(I0, N, platform, t_max)
            histories[platform] = history
            
            # Calculate statistics
            peak_time, peak_infections = self.find_peak()
            final_recovered = history['R'].iloc[-1]
            
            results[platform] = {
                'R0': self.calculate_R0(),  
                'peak_time': peak_time,
                'peak_infections': peak_infections,
                'total_infected_pct': (final_recovered / N) * 100,
                'half_life': self._calculate_half_life(history)
            }
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        for platform, history in histories.items():
            plt.plot(history['t'], history['I'], label=f"{platform.capitalize()}", linewidth=3)
        
        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('Active Users', fontsize=14)
        plt.title('Linguistic Innovation Spread Across Platforms', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add annotations
        for i, platform in enumerate(platforms):
            stats = results[platform]
            y_pos = 0.9 - i * 0.15
            stats_text = (f"{platform.capitalize()}: R₀={stats['R0']:.2f}, "
                         f"Peak={stats['peak_infections']:.0f} at t={stats['peak_time']:.1f}")
            
            plt.text(0.02, y_pos, stats_text, transform=plt.gca().transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return plt.gcf(), pd.DataFrame(results).T
    
    def _calculate_half_life(self, history):
        """Calculate the half-life of the linguistic innovation"""
        peak_idx = history['I'].idxmax()
        peak_value = history['I'].iloc[peak_idx]
        
        # Find when it drops to half the peak after the peak
        post_peak = history.iloc[peak_idx:]
        half_points = post_peak[post_peak['I'] <= peak_value/2]
        
        if not half_points.empty:
            half_life = half_points.iloc[0]['t'] - history.iloc[peak_idx]['t']
            return half_life
        else:
            return None  # Hasn't reached half-life yet


def main():
    """
    Test the NetworkLCM model with both Twitter and Reddit networks.
    Compare results with standard SIR model.
    """
    print("=" * 80)
    print("TESTING NETWORK LINGUISTIC CONTAGION MODEL")
    print("=" * 80)
    
    # Simulation parameters
    N = 5000        # Network size
    I0 = 10         # Initial adopters
    t_max = 100     # Simulation duration
    
    # Initialize both models with the same parameters
    base_params = SIRParameters(
        beta=0.3,    # Transmission rate
        alpha=0.1,   # Recovery rate
        nu=0.05,     # Mutation rate
        mu=0.02,     # Birth rate
        gamma=0.01   # Re-susceptibility rate
    )
    
    # Standard SIR model
    print("\n1. Running standard SIR model (no network effects)...")
    std_model = LinguisticContagionModel(params=base_params)
    std_history = std_model.simulate(I0, N, t_max)
    std_stats = std_model.summary_statistics()
    
    print(f"  Standard SIR Results:")
    print(f"  - R₀: {std_stats['R0']:.2f}")
    print(f"  - Peak Time: Day {std_stats['peak_time']:.1f}")
    print(f"  - Peak Users: {std_stats['peak_infections']:.0f}")
    print(f"  - Total Reached: {std_stats['total_infected_pct']:.1f}%")
    
    # Network-aware SIR model for Twitter
    print("\n2. Running network SIR model for Twitter...")
    twitter_model = NetworkLCM(params=base_params)
    twitter_history = twitter_model.simulate_network_spread(I0, N, platform="twitter", t_max=t_max)
    twitter_stats = twitter_model.summary_statistics()
    
    print(f"  Twitter Network Results:")
    print(f"  - R₀: {twitter_stats['R0']:.2f}")
    print(f"  - Peak Time: Day {twitter_stats['peak_time']:.1f}")
    print(f"  - Peak Users: {twitter_stats['peak_infections']:.0f}")
    print(f"  - Total Reached: {twitter_stats['total_infected_pct']:.1f}%")
    
    # Network-aware SIR model for Reddit
    print("\n3. Running network SIR model for Reddit...")
    reddit_model = NetworkLCM(params=base_params)
    reddit_history = reddit_model.simulate_network_spread(I0, N, platform="reddit", t_max=t_max)
    reddit_stats = reddit_model.summary_statistics()
    
    print(f"  Reddit Network Results:")
    print(f"  - R₀: {reddit_stats['R0']:.2f}")
    print(f"  - Peak Time: Day {reddit_stats['peak_time']:.1f}")
    print(f"  - Peak Users: {reddit_stats['peak_infections']:.0f}")
    print(f"  - Total Reached: {reddit_stats['total_infected_pct']:.1f}%")
    
    # Compare all models
    print("\n4. Comparing spread dynamics across models...")
    plt.figure(figsize=(12, 8))
    
    plt.plot(std_history['t'], std_history['I'], 'k-', 
             label='Standard SIR (Homogeneous)', linewidth=2)
    plt.plot(twitter_history['t'], twitter_history['I'], 'b-', 
             label='Twitter Network', linewidth=2)
    plt.plot(reddit_history['t'], reddit_history['I'], 'r-', 
             label='Reddit Network', linewidth=2)
    
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Active Users', fontsize=14)
    plt.title('Linguistic Innovation Spread: Network Effects', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Visualization of network state
    print("\n5. Visualizing network state for each platform...")
    twitter_vis = twitter_model.visualize_network_state()
    reddit_vis = reddit_model.visualize_network_state()
    
    # More advanced comparison (optional)
    print("\n6. Simulating the spread of multiple linguistic innovations...")
    # Simulate different "innovations" by varying initial parameters
    scenarios = {
        "Slang term": SIRParameters(beta=0.3, alpha=0.1, nu=0.05, mu=0.02, gamma=0.01),
        "Meme/joke": SIRParameters(beta=0.5, alpha=0.2, nu=0.1, mu=0.02, gamma=0.01),  # Fast spread, fast decay
        "Political term": SIRParameters(beta=0.25, alpha=0.05, nu=0.1, mu=0.03, gamma=0.005)  # Polarizing, mutates more
    }
    
    # Compare between Twitter and Reddit
    results = {}
    for platform in ["twitter", "reddit"]:
        platform_results = {}
        for name, params in scenarios.items():
            model = NetworkLCM(params=params)
            model.simulate_network_spread(I0, N, platform=platform, t_max=t_max)
            stats = model.summary_statistics()
            platform_results[name] = {
                'R0': stats['R0'],
                'peak_time': stats['peak_time'],
                'peak_users': stats['peak_infections'],
                'total_pct': stats['total_infected_pct']
            }
        results[platform] = platform_results
    
    # Display comparative results
    twitter_df = pd.DataFrame(results['twitter']).T
    reddit_df = pd.DataFrame(results['reddit']).T
    
    print("\nTwitter results for different linguistic innovations:")
    print(twitter_df)
    print("\nReddit results for different linguistic innovations:")
    print(reddit_df)
    
    print("\nAnalysis complete! Displaying visualizations...")
    plt.show()
    
    return std_model, twitter_model, reddit_model, results


if __name__ == "__main__":
    std_model, twitter_model, reddit_model, results = main()