#!/usr/bin/env python3
"""
NBA Props Model - Enhanced Analysis Demo
Demonstrates key analyses with immediate value for betting insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class NBAPropsAnalyzer:
    """Complete NBA props analysis pipeline"""
    
    def __init__(self, data_path='/Users/diyagamah/Documents/nba_props_model/data'):
        self.data_path = Path(data_path)
        self.player_df = None
        self.risk_reward = None
        
    def load_data(self):
        """Load and merge all CTG data"""
        print("Loading CTG data...")
        
        # Define file paths
        base_path = self.data_path / 'ctg_data_organized' / 'players' / '2023-24' / 'regular_season'
        
        # Load main datasets
        try:
            offensive = pd.read_csv(base_path / 'offensive_overview' / 'offensive_overview.csv')
            defense = pd.read_csv(base_path / 'defense_rebounding' / 'defense_rebounding.csv')
            shooting = pd.read_csv(base_path / 'shooting_overall' / 'shooting_overall.csv')
            fouls = pd.read_csv(base_path / 'foul_drawing' / 'foul_drawing.csv')
            
            # Merge datasets
            self.player_df = offensive.copy()
            
            # Add defense stats
            defense_cols = ['Player', 'Team', 'fgOR%', 'fgDR%', 'ftOR%', 'ftDR%', 'BLK%', 'STL%']
            self.player_df = self.player_df.merge(
                defense[defense_cols], 
                on=['Player', 'Team'], 
                how='left'
            )
            
            # Add shooting stats
            shooting_cols = ['Player', 'Team', 'eFG%', '2P%', '3P%', 'FT%']
            self.player_df = self.player_df.merge(
                shooting[shooting_cols],
                on=['Player', 'Team'],
                how='left'
            )
            
            # Add foul drawing
            foul_cols = ['Player', 'Team', 'SFLD%', 'FFLD%', 'AND1%']
            self.player_df = self.player_df.merge(
                fouls[foul_cols],
                on=['Player', 'Team'],
                how='left'
            )
            
            print(f"✓ Loaded {len(self.player_df)} players with {len(self.player_df.columns)} features")
            return self.player_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_pra_estimates(self):
        """Create PRA estimates from available stats"""
        if self.player_df is None:
            return None
            
        print("\nCreating PRA estimates...")
        
        # Points estimate
        if all(col in self.player_df.columns for col in ['MIN', 'Usage', 'PSA']):
            points = self.player_df['MIN'] * (self.player_df['Usage'] / 100) * self.player_df['PSA'] * 2
        else:
            points = 15  # Default
            
        # Rebounds estimate
        if all(col in self.player_df.columns for col in ['MIN', 'fgDR%', 'fgOR%']):
            rebounds = self.player_df['MIN'] * ((self.player_df['fgDR%'] + self.player_df['fgOR%']) / 100) * 4
        else:
            rebounds = 5  # Default
            
        # Assists estimate
        if all(col in self.player_df.columns for col in ['MIN', 'AST%']):
            assists = self.player_df['MIN'] * (self.player_df['AST%'] / 100) * 3
        else:
            assists = 3  # Default
            
        self.player_df['PRA_estimate'] = points + rebounds + assists
        
        print(f"✓ Created PRA estimates (mean: {self.player_df['PRA_estimate'].mean():.1f})")
        return self.player_df['PRA_estimate']
    
    def calculate_volatility_scores(self):
        """Calculate player volatility/consistency scores"""
        if self.player_df is None:
            return None
            
        print("\nCalculating volatility scores...")
        
        volatility_factors = []
        
        # Factor 1: Minutes volatility (low minutes = high volatility)
        if 'MIN' in self.player_df.columns:
            min_factor = 1 - (self.player_df['MIN'] / self.player_df['MIN'].max())
            volatility_factors.append(min_factor)
        
        # Factor 2: Usage volatility (middle usage = higher volatility)
        if 'Usage' in self.player_df.columns:
            usage_norm = self.player_df['Usage'] / self.player_df['Usage'].max()
            usage_factor = 1 - np.abs(usage_norm - 0.5) * 2
            volatility_factors.append(usage_factor)
        
        # Factor 3: Turnover volatility
        if 'TOV%' in self.player_df.columns:
            tov_factor = self.player_df['TOV%'].fillna(0) / 20  # Normalize to ~0-1
            volatility_factors.append(tov_factor)
        
        # Combine factors
        if volatility_factors:
            self.player_df['Volatility_Score'] = np.mean(volatility_factors, axis=0)
            self.player_df['Volatility_Rating'] = pd.cut(
                self.player_df['Volatility_Score'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            print(f"✓ Calculated volatility scores")
        
        return self.player_df['Volatility_Score'] if 'Volatility_Score' in self.player_df.columns else None
    
    def create_risk_reward_matrix(self):
        """Create betting categories based on risk/reward"""
        if self.player_df is None:
            return None
            
        print("\nCreating risk-reward matrix...")
        
        # Ensure we have required columns
        if 'PRA_estimate' not in self.player_df.columns:
            self.create_pra_estimates()
        if 'Volatility_Score' not in self.player_df.columns:
            self.calculate_volatility_scores()
        
        # Filter for qualified players
        qualified = self.player_df[self.player_df['MIN'] >= 20].copy()
        
        # Create quadrants
        median_pra = qualified['PRA_estimate'].median()
        median_volatility = qualified['Volatility_Score'].median()
        
        conditions = [
            (qualified['PRA_estimate'] >= median_pra) & (qualified['Volatility_Score'] < median_volatility),
            (qualified['PRA_estimate'] >= median_pra) & (qualified['Volatility_Score'] >= median_volatility),
            (qualified['PRA_estimate'] < median_pra) & (qualified['Volatility_Score'] < median_volatility),
            (qualified['PRA_estimate'] < median_pra) & (qualified['Volatility_Score'] >= median_volatility)
        ]
        
        choices = ['Premium Plays', 'High Risk/Reward', 'Safe Unders', 'Avoid']
        qualified['Betting_Category'] = np.select(conditions, choices, default='Unknown')
        
        self.risk_reward = qualified
        
        print(f"✓ Created risk-reward categories:")
        print(qualified['Betting_Category'].value_counts())
        
        return self.risk_reward
    
    def identify_player_clusters(self, n_clusters=6):
        """Cluster players into archetypes"""
        if self.player_df is None:
            return None
            
        print(f"\nIdentifying {n_clusters} player archetypes...")
        
        # Select clustering features
        feature_cols = ['Usage', 'PSA', 'AST%', 'fgDR%', 'BLK%', 'STL%', 'eFG%', '3P%', 'MIN']
        available_features = [col for col in feature_cols if col in self.player_df.columns]
        
        # Filter qualified players
        qualified = self.player_df[self.player_df['MIN'] > 15].copy()
        
        # Prepare features
        X = qualified[available_features].fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        qualified['Cluster'] = clusters
        
        # Name clusters based on characteristics
        cluster_names = {}
        for i in range(n_clusters):
            cluster_data = qualified[qualified['Cluster'] == i]
            
            # Determine archetype
            avg_usage = cluster_data['Usage'].mean()
            avg_ast = cluster_data['AST%'].mean() if 'AST%' in cluster_data.columns else 0
            avg_reb = cluster_data['fgDR%'].mean() if 'fgDR%' in cluster_data.columns else 0
            avg_3p = cluster_data['3P%'].mean() if '3P%' in cluster_data.columns else 0
            
            if avg_usage > 25:
                name = "Primary Scorer"
            elif avg_ast > 20:
                name = "Playmaker"
            elif avg_reb > 15:
                name = "Elite Rebounder"
            elif avg_3p > 0.38:
                name = "Sharpshooter"
            elif cluster_data['MIN'].mean() < 20:
                name = "Bench Player"
            else:
                name = "Role Player"
            
            cluster_names[i] = name
        
        qualified['Archetype'] = qualified['Cluster'].map(cluster_names)
        
        print(f"✓ Identified player archetypes:")
        print(qualified['Archetype'].value_counts())
        
        return qualified
    
    def generate_betting_sheet(self):
        """Generate actionable betting recommendations"""
        if self.risk_reward is None:
            self.create_risk_reward_matrix()
            
        print("\nGenerating betting recommendations...")
        
        recommendations = []
        
        # Premium plays
        premium = self.risk_reward[
            self.risk_reward['Betting_Category'] == 'Premium Plays'
        ].nlargest(5, 'PRA_estimate')
        
        for _, player in premium.iterrows():
            recommendations.append({
                'Player': player['Player'],
                'Team': player['Team'],
                'Category': 'Premium',
                'PRA_Projection': round(player['PRA_estimate'], 1),
                'Volatility': player['Volatility_Rating'],
                'Confidence': 'HIGH'
            })
        
        # High risk/reward
        high_risk = self.risk_reward[
            self.risk_reward['Betting_Category'] == 'High Risk/Reward'
        ].nlargest(3, 'PRA_estimate')
        
        for _, player in high_risk.iterrows():
            recommendations.append({
                'Player': player['Player'],
                'Team': player['Team'],
                'Category': 'High Risk',
                'PRA_Projection': round(player['PRA_estimate'], 1),
                'Volatility': player['Volatility_Rating'],
                'Confidence': 'MEDIUM'
            })
        
        betting_sheet = pd.DataFrame(recommendations)
        
        print(f"✓ Generated {len(betting_sheet)} betting recommendations")
        
        return betting_sheet
    
    def visualize_insights(self):
        """Create key visualizations"""
        if self.risk_reward is None:
            self.create_risk_reward_matrix()
            
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk-Reward Scatter
        ax1 = axes[0, 0]
        categories = self.risk_reward['Betting_Category'].unique()
        colors = {'Premium Plays': 'green', 'High Risk/Reward': 'orange', 
                 'Safe Unders': 'blue', 'Avoid': 'red'}
        
        for category in categories:
            if category in colors:
                data = self.risk_reward[self.risk_reward['Betting_Category'] == category]
                ax1.scatter(data['Volatility_Score'], data['PRA_estimate'], 
                          label=category, alpha=0.6, color=colors.get(category, 'gray'))
        
        ax1.axhline(y=self.risk_reward['PRA_estimate'].median(), color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.risk_reward['Volatility_Score'].median(), color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Expected PRA (Reward)')
        ax1.set_title('Risk-Reward Matrix')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top PRA Projections
        ax2 = axes[0, 1]
        top_pra = self.risk_reward.nlargest(15, 'PRA_estimate')
        ax2.barh(range(len(top_pra)), top_pra['PRA_estimate'].values)
        ax2.set_yticks(range(len(top_pra)))
        ax2.set_yticklabels([p.split()[-1] for p in top_pra['Player'].values])  # Last name only
        ax2.set_xlabel('Expected PRA')
        ax2.set_title('Top 15 PRA Projections')
        ax2.grid(True, alpha=0.3)
        
        # 3. Volatility Distribution
        ax3 = axes[1, 0]
        if 'Volatility_Rating' in self.risk_reward.columns:
            volatility_counts = self.risk_reward['Volatility_Rating'].value_counts()
            ax3.pie(volatility_counts.values, labels=volatility_counts.index, autopct='%1.1f%%',
                   colors=['green', 'orange', 'red'])
            ax3.set_title('Player Volatility Distribution')
        
        # 4. Usage vs PRA
        ax4 = axes[1, 1]
        ax4.scatter(self.risk_reward['Usage'], self.risk_reward['PRA_estimate'], 
                   c=self.risk_reward['MIN'], cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Usage Rate (%)')
        ax4.set_ylabel('Expected PRA')
        ax4.set_title('Usage vs PRA (colored by minutes)')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Minutes')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('NBA Props Analysis Dashboard', y=1.02, fontsize=16, fontweight='bold')
        
        return fig
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("NBA PROPS MODEL - ENHANCED ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        if self.player_df is not None:
            # Create estimates
            self.create_pra_estimates()
            
            # Calculate volatility
            self.calculate_volatility_scores()
            
            # Create risk-reward matrix
            self.create_risk_reward_matrix()
            
            # Identify clusters
            clusters = self.identify_player_clusters()
            
            # Generate betting sheet
            betting_sheet = self.generate_betting_sheet()
            
            print("\n" + "="*60)
            print("TOP BETTING RECOMMENDATIONS:")
            print("="*60)
            print(betting_sheet.to_string(index=False))
            
            # Create visualizations
            fig = self.visualize_insights()
            
            # Save outputs
            output_path = Path('/Users/diyagamah/Documents/nba_props_model/outputs')
            output_path.mkdir(exist_ok=True)
            
            betting_sheet.to_csv(output_path / 'betting_recommendations.csv', index=False)
            self.risk_reward.to_csv(output_path / 'risk_reward_analysis.csv', index=False)
            
            if fig:
                fig.savefig(output_path / 'analysis_dashboard.png', dpi=100, bbox_inches='tight')
                print(f"\n✓ Saved visualization to {output_path / 'analysis_dashboard.png'}")
            
            print(f"✓ Saved betting sheet to {output_path / 'betting_recommendations.csv'}")
            print(f"✓ Saved risk-reward analysis to {output_path / 'risk_reward_analysis.csv'}")
            
            # Show plot
            plt.show()
            
            return betting_sheet, self.risk_reward
        
        return None, None


def main():
    """Main execution function"""
    analyzer = NBAPropsAnalyzer()
    betting_sheet, risk_reward = analyzer.run_complete_analysis()
    
    if betting_sheet is not None:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nKey Outputs:")
        print(f"- {len(betting_sheet)} betting recommendations generated")
        print(f"- {len(risk_reward)} players analyzed")
        print(f"- Premium plays: {(risk_reward['Betting_Category'] == 'Premium Plays').sum()}")
        print(f"- High risk/reward: {(risk_reward['Betting_Category'] == 'High Risk/Reward').sum()}")
    else:
        print("\nAnalysis failed. Please check data files.")


if __name__ == "__main__":
    main()