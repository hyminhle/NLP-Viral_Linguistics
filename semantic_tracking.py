"""
1. Track how term meanings shift as they spread across communities
2. Detect semantic drift over time using dynamic embeddings
3. Build Semantic Genealogy Graphs showing meaning evolution
4. Identify "semantic mutations" as terms spread virally

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer


@dataclass
class SemanticSnapshot:
    """Represents meaning of a term at a specific time point"""
    timestamp: datetime
    embedding: np.ndarray
    context_examples: List[str]
    community: str
    usage_count: int


class SemanticTracker:
    """
    Tracks semantic evolution of linguistic terms over time.
    
    Uses sentence embeddings to capture meaning and tracks how
    meanings drift as terms spread through communities.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize semantic tracker.
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.term_histories: Dict[str, List[SemanticSnapshot]] = {}
        
    def track_term_over_time(self, 
                            df: pd.DataFrame, 
                            term: str,
                            text_col: str = 'text',
                            time_col: str = 'time',
                            community_col: str = 'meta',
                            time_bins: int = 10) -> List[SemanticSnapshot]:
        """
        Track semantic evolution of a term over time.
        
        Args:
            df: DataFrame with text data
            term: Term to track
            text_col: Column with text content
            time_col: Column with timestamps
            community_col: Column with community labels
            time_bins: Number of time periods to analyze
            
        Returns:
            List of SemanticSnapshot objects representing evolution
        """
        # Convert time to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], unit='s')
        
        # Filter to rows containing the term
        mask = df[text_col].str.contains(term, case=False, na=False)
        term_df = df[mask].copy()
        
        if len(term_df) == 0:
            print(f"⚠️ No instances of '{term}' found in dataset")
            return []
        
        print(f"📊 Found {len(term_df)} instances of '{term}'")
        
        # Create time bins
        term_df['time_bin'] = pd.cut(term_df[time_col], bins=time_bins)
        
        snapshots = []
        
        for time_bin in term_df['time_bin'].cat.categories:
            bin_data = term_df[term_df['time_bin'] == time_bin]
            
            if len(bin_data) == 0:
                continue
            
            # Get context sentences (extract sentences containing term)
            contexts = []
            for text in bin_data[text_col].head(50):  # Sample up to 50
                # Simple sentence extraction
                sentences = [s.strip() for s in str(text).split('.') if term.lower() in s.lower()]
                contexts.extend(sentences[:3])  # Max 3 per text
            
            if not contexts:
                continue
            
            # Create composite context for embedding
            composite_context = f"{term} || " + " | ".join(contexts[:10])
            
            # Generate embedding
            embedding = self.model.encode([composite_context], 
                                         normalize_embeddings=True)[0]
            
            # Get dominant community
            dominant_community = bin_data[community_col].mode()[0] if len(bin_data) > 0 else 'unknown'
            
            snapshot = SemanticSnapshot(
                timestamp=bin_data[time_col].mean(),
                embedding=embedding,
                context_examples=contexts[:5],
                community=dominant_community,
                usage_count=len(bin_data)
            )
            
            snapshots.append(snapshot)
        
        self.term_histories[term] = snapshots
        return snapshots
    
    def calculate_semantic_drift(self, term: str) -> Dict:
        """
        Calculate semantic drift metrics for a term.
        
        Args:
            term: Term to analyze
            
        Returns:
            Dictionary with drift metrics
        """
        if term not in self.term_histories:
            raise ValueError(f"Term '{term}' not tracked. Run track_term_over_time first.")
        
        snapshots = self.term_histories[term]
        
        if len(snapshots) < 2:
            return {'drift_detected': False, 'reason': 'Insufficient time points'}
        
        # Calculate pairwise similarities
        embeddings = np.array([s.embedding for s in snapshots])
        similarities = cosine_similarity(embeddings)
        
        # Drift metrics
        drift_scores = []
        for i in range(len(snapshots) - 1):
            drift = 1 - similarities[i, i+1]  # Dissimilarity
            drift_scores.append(drift)
        
        # Overall drift: first to last
        total_drift = 1 - similarities[0, -1]
        
        # Velocity: average drift per time step
        drift_velocity = np.mean(drift_scores) if drift_scores else 0
        
        # Acceleration: change in drift rate
        if len(drift_scores) > 1:
            drift_acceleration = np.diff(drift_scores).mean()
        else:
            drift_acceleration = 0
        
        results = {
            'term': term,
            'total_drift': float(total_drift),
            'drift_velocity': float(drift_velocity),
            'drift_acceleration': float(drift_acceleration),
            'max_drift': float(max(drift_scores)) if drift_scores else 0,
            'drift_detected': total_drift > 0.15,  # Threshold
            'time_points': len(snapshots),
            'temporal_drift_scores': drift_scores
        }
        
        return results
    
    def detect_semantic_mutations(self, term: str, mutation_threshold: float = 0.2) -> List[Tuple[int, float]]:
        """
        Detect significant semantic shifts ("mutations").
        
        Args:
            term: Term to analyze
            mutation_threshold: Minimum drift to consider a mutation
            
        Returns:
            List of (time_point_index, drift_score) for mutations
        """
        if term not in self.term_histories:
            raise ValueError(f"Term '{term}' not tracked")
        
        snapshots = self.term_histories[term]
        embeddings = np.array([s.embedding for s in snapshots])
        
        mutations = []
        for i in range(len(snapshots) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0, 0]
            drift = 1 - similarity
            
            if drift > mutation_threshold:
                mutations.append((i, drift))
        
        return mutations
    
    def compare_communities(self, term: str) -> Dict:
        """
        Compare how term meaning differs across communities.
        
        Args:
            term: Term to analyze
            
        Returns:
            Dictionary with community comparison metrics
        """
        if term not in self.term_histories:
            raise ValueError(f"Term '{term}' not tracked")
        
        snapshots = self.term_histories[term]
        
        # Group by community
        community_embeddings = {}
        for snapshot in snapshots:
            if snapshot.community not in community_embeddings:
                community_embeddings[snapshot.community] = []
            community_embeddings[snapshot.community].append(snapshot.embedding)
        
        # Average embedding per community
        community_means = {}
        for community, embeddings in community_embeddings.items():
            community_means[community] = np.mean(embeddings, axis=0)
        
        # Calculate pairwise differences
        communities = list(community_means.keys())
        n_communities = len(communities)
        
        divergence_matrix = np.zeros((n_communities, n_communities))
        
        for i in range(n_communities):
            for j in range(n_communities):
                if i != j:
                    sim = cosine_similarity([community_means[communities[i]]], 
                                           [community_means[communities[j]]])[0, 0]
                    divergence_matrix[i, j] = 1 - sim
        
        return {
            'communities': communities,
            'divergence_matrix': divergence_matrix,
            'max_divergence': float(np.max(divergence_matrix)),
            'mean_divergence': float(np.mean(divergence_matrix[divergence_matrix > 0]))
        }


class SemanticGenealogyGraph:
    """
    Build and analyze Semantic Genealogy Graphs showing meaning evolution.
    
    Nodes represent different meanings, edges represent evolutionary relationships.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        
    def build_from_tracker(self, tracker: SemanticTracker, term: str, 
                          cluster_threshold: float = 0.15) -> nx.DiGraph:
        """
        Build genealogy graph from semantic tracker data.
        
        Args:
            tracker: SemanticTracker with term history
            term: Term to build graph for
            cluster_threshold: Distance threshold for clustering meanings
            
        Returns:
            NetworkX directed graph
        """
        if term not in tracker.term_histories:
            raise ValueError(f"Term '{term}' not tracked")
        
        snapshots = tracker.term_histories[term]
        
        if len(snapshots) < 2:
            print("⚠️ Not enough data points to build genealogy")
            return self.graph
        
        # Extract embeddings and metadata
        embeddings = np.array([s.embedding for s in snapshots])
        
        # Cluster similar meanings using DBSCAN
        clustering = DBSCAN(eps=cluster_threshold, min_samples=1, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Create nodes for each meaning cluster
        unique_labels = np.unique(labels)
        cluster_nodes = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_snapshots = [s for i, s in enumerate(snapshots) if mask[i]]
            
            # Node attributes
            node_id = f"meaning_{self.node_counter}"
            self.node_counter += 1
            
            # Representative examples
            all_examples = []
            for s in cluster_snapshots:
                all_examples.extend(s.context_examples)
            
            self.graph.add_node(node_id, 
                              label=f"M{label}",
                              time_range=(min(s.timestamp for s in cluster_snapshots),
                                        max(s.timestamp for s in cluster_snapshots)),
                              examples=all_examples[:3],
                              usage_count=sum(s.usage_count for s in cluster_snapshots))
            
            cluster_nodes[label] = node_id
        
        # Create edges based on temporal evolution
        for i in range(len(snapshots) - 1):
            source_label = labels[i]
            target_label = labels[i + 1]
            
            if source_label != target_label:
                source_node = cluster_nodes[source_label]
                target_node = cluster_nodes[target_label]
                
                # Calculate transition strength
                similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0, 0]
                
                if self.graph.has_edge(source_node, target_node):
                    # Strengthen existing edge
                    self.graph[source_node][target_node]['weight'] += 1
                else:
                    self.graph.add_edge(source_node, target_node,
                                      weight=1,
                                      similarity=similarity)
        
        return self.graph
    
    def identify_semantic_bridges(self) -> List[str]:
        """
        Identify meaning nodes that bridge different semantic clusters.
        
        Returns:
            List of bridge node IDs
        """
        if len(self.graph.nodes) == 0:
            return []
        
        # Nodes with high betweenness centrality are bridges
        centrality = nx.betweenness_centrality(self.graph)
        
        # Top quartile
        threshold = np.percentile(list(centrality.values()), 75)
        bridges = [node for node, score in centrality.items() if score >= threshold]
        
        return bridges
    
    def plot_genealogy(self, term: str, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize semantic genealogy graph.
        
        Args:
            term: Term being visualized
            figsize: Figure size
        """
        if len(self.graph.nodes) == 0:
            print("⚠️ Graph is empty")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Node sizes based on usage
        node_sizes = [self.graph.nodes[node].get('usage_count', 100) 
                     for node in self.graph.nodes]
        node_sizes = [max(300, min(3000, s)) for s in node_sizes]  # Normalize
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_size=node_sizes,
                              node_color='lightblue',
                              edgecolors='darkblue',
                              linewidths=2,
                              ax=ax)
        
        # Draw edges with varying thickness
        edges = self.graph.edges()
        weights = [self.graph[u][v].get('weight', 1) for u, v in edges]
        
        nx.draw_networkx_edges(self.graph, pos,
                              width=[w * 2 for w in weights],
                              alpha=0.6,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              ax=ax)
        
        # Labels
        labels = {node: self.graph.nodes[node].get('label', node) 
                 for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, 
                               font_size=12, font_weight='bold',
                               ax=ax)
        
        ax.set_title(f'Semantic Genealogy Graph: "{term}"', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return fig


def visualize_semantic_drift(tracker: SemanticTracker, term: str, 
                             figsize: Tuple[int, int] = (14, 10)):
    """
    Create comprehensive visualization of semantic drift.
    
    Args:
        tracker: SemanticTracker with term data
        term: Term to visualize
        figsize: Figure size
    """
    if term not in tracker.term_histories:
        print(f"⚠️ Term '{term}' not tracked")
        return
    
    snapshots = tracker.term_histories[term]
    
    if len(snapshots) < 2:
        print("⚠️ Need at least 2 time points")
        return
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Drift over time
    ax1 = fig.add_subplot(gs[0, :])
    
    embeddings = np.array([s.embedding for s in snapshots])
    times = [s.timestamp for s in snapshots]
    
    drift_scores = []
    drift_times = []
    for i in range(len(snapshots) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0, 0]
        drift = 1 - sim
        drift_scores.append(drift)
        drift_times.append(times[i])
    
    ax1.plot(drift_times, drift_scores, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.axhline(y=0.15, color='gray', linestyle='--', label='Mutation Threshold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Semantic Drift', fontsize=12)
    ax1.set_title(f'Temporal Semantic Drift: "{term}"', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Similarity heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    
    similarity_matrix = cosine_similarity(embeddings)
    
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, square=True, cbar_kws={'label': 'Cosine Similarity'},
                ax=ax2)
    ax2.set_title('Cross-Time Similarity Matrix', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Point', fontsize=10)
    ax2.set_ylabel('Time Point', fontsize=10)
    
    # 3. Community distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    communities = [s.community for s in snapshots]
    usage_counts = [s.usage_count for s in snapshots]
    
    community_totals = {}
    for comm, count in zip(communities, usage_counts):
        community_totals[comm] = community_totals.get(comm, 0) + count
    
    ax3.bar(community_totals.keys(), community_totals.values(), color='skyblue', edgecolor='navy')
    ax3.set_xlabel('Community', fontsize=12)
    ax3.set_ylabel('Total Usage', fontsize=12)
    ax3.set_title('Usage Across Communities', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Drift statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    drift_metrics = tracker.calculate_semantic_drift(term)
    
    stats_text = f"""
    SEMANTIC DRIFT ANALYSIS
    {'=' * 60}
    
    Total Drift (First → Last):        {drift_metrics['total_drift']:.4f}
    Average Drift Velocity:             {drift_metrics['drift_velocity']:.4f}
    Maximum Single-Step Drift:          {drift_metrics['max_drift']:.4f}
    Drift Acceleration:                 {drift_metrics['drift_acceleration']:.4f}
    
    Time Points Analyzed:               {drift_metrics['time_points']}
    Mutation Detected:                  {'✓ YES' if drift_metrics['drift_detected'] else '✗ NO'}
    
    {'=' * 60}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Comprehensive Semantic Analysis: "{term}"', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def analyze_multiple_terms(tracker: SemanticTracker, 
                          terms: List[str]) -> pd.DataFrame:
    """
    Compare semantic drift across multiple terms.
    
    Args:
        tracker: SemanticTracker with term data
        terms: List of terms to compare
        
    Returns:
        DataFrame with comparative metrics
    """
    results = []
    
    for term in terms:
        if term not in tracker.term_histories:
            print(f"⚠️ Skipping '{term}' - not tracked")
            continue
        
        metrics = tracker.calculate_semantic_drift(term)
        mutations = tracker.detect_semantic_mutations(term)
        
        results.append({
            'term': term,
            'total_drift': metrics['total_drift'],
            'drift_velocity': metrics['drift_velocity'],
            'max_drift': metrics['max_drift'],
            'mutation_count': len(mutations),
            'time_points': metrics['time_points'],
            'drift_detected': metrics['drift_detected']
        })
    
    return pd.DataFrame(results).sort_values('total_drift', ascending=False)


if __name__ == "__main__":
    print("=" * 70)
    print("SEMANTIC TRACKING MODULE - Viral Linguistics Project")
    print("=" * 70)
    print("\nThis module provides:")
    print("  ✓ Semantic drift detection using embeddings")
    print("  ✓ Temporal meaning evolution tracking")
    print("  ✓ Semantic Genealogy Graph construction")
    print("  ✓ Cross-community semantic comparison")
    print("  ✓ Mutation detection for viral terms")
    print("\nReady for integration with SIR model and data preprocessing.")