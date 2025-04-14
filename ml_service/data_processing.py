import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
def combine_csv_files(raw_data_path, processed_data_path):
    """
    Load CSV files from the raw_data_path, combine them into a single DataFrame,
    perform basic cleaning, add a city column, and save the combined DataFrame.
    """
    # Ensure the processed directory exists
    os.makedirs(processed_data_path, exist_ok=True)

    # Define file names for each city
    file_names = {
        'Oakville': 'OakvillePrice.csv',
        'Guelph': 'GuelphPrice.csv',
        'Mississauga': 'MissisaugaPrice.csv',
        'Milton': 'MiltonPrice.csv'
    }
    
    # Load the CSV files and convert the price column to float64
    dataframes = {}
    for city, file in file_names.items():
        file_path = os.path.join(raw_data_path, file)
        df = pd.read_csv(file_path)
        df['price'] = df['price'].astype('float64')
        dataframes[city] = df
    
    # Combine all dataframes into one DataFrame
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    
    # Assign city names based on the original datasets
    city_sequence = []
    for city, df in dataframes.items():
        city_sequence.extend([city] * len(df))
    combined_df['city'] = city_sequence

    # Fill missing 'price' values with the city mean
    for city in combined_df['city'].unique():
        city_mean = combined_df.loc[combined_df['city'] == city, 'price'].mean()
        mask = (combined_df['city'] == city) & (combined_df['price'].isnull())
        combined_df.loc[mask, 'price'] = city_mean

    # Drop columns that are completely empty
    columns_to_drop = combined_df.columns[combined_df.isna().all()].tolist()
    if columns_to_drop:
        print(f"Dropping completely empty columns: {columns_to_drop}")
        combined_df.drop(columns=columns_to_drop, inplace=True)

    # Save the combined dataframe to CSV
    output_file = os.path.join(processed_data_path, 'combined_properties.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataframe saved to: {output_file}")

    # Display basic information about the combined dataframe
    print("=" * 50)
    print(f"Total number of rows: {len(combined_df)}")
    print(f"Total number of columns: {len(combined_df.columns)}")
    print("\nFirst 5 rows:")
    print(combined_df.head())
    print("\nMissing values per column:")
    print(combined_df.isnull().sum())
    print("\nData types of columns:")
    print(combined_df.dtypes)
    
    return combined_df

def load_data(file_path='combined_properties.csv'):
    """
    Load and preprocess the combined real estate data.
    This includes handling missing values, filtering out extreme values,
    and creating new features.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Fill missing values for key features
    df['bed'].fillna(df['bed'].median(), inplace=True)
    df['bath'].fillna(df['bath'].median(), inplace=True)
    
    # Filter out extreme price values
    df = df[(df['price'] > 100000) & (df['price'] < 10000000)]
    
    # Create new feature: bed_bath_ratio
    df['bed_bath_ratio'] = df['bed'] / df['bath']
    
    # Estimate missing 'sqft' values based on city and price
    if df['sqft'].isna().sum() > 0:
        city_price_per_sqft = df.groupby('city')['pricePerSf'].median().to_dict()
        for city, price_per_sqft in city_price_per_sqft.items():
            if not np.isnan(price_per_sqft):
                mask = (df['city'] == city) & (df['sqft'].isna())
                df.loc[mask, 'sqft'] = df.loc[mask, 'price'] / price_per_sqft

    # Define features and target
    features = ['city', 'bed', 'bath', 'sqft', 'lotArea', 'homeType', 'bed_bath_ratio']
    target = 'price'
    # Keep only rows where all selected features and the target are available
    modeling_df = df[features + [target]].dropna()
    print(f"After preprocessing, dataset shape: {modeling_df.shape}")
    
    return modeling_df, features, target

def visualize_data(df):
    """
    Generate visualizations of the dataset and save them as image files.
    Currently creates a boxplot of house prices by city.
    """
     # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    print("Generating visualizations...")
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Define a money formatter for price axes
    def money_formatter(x, pos):
        if x >= 1e6:
            return f'${x/1e6:.1f}M'
        else:
            return f'${x/1e3:.0f}K'
    
    money_format = FuncFormatter(money_formatter)
    
    # 1. Price distribution by city (boxplot)
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='city', y='price', data=df, palette='viridis')
    plt.title('House Price Distribution by City')
    plt.xlabel('City')
    plt.ylabel('Price')
    ax.yaxis.set_major_formatter(money_format)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/price_by_city_boxplot.png', dpi=300)
    plt.close()
    
    # 2. Price vs Square Footage with City color (scatter plot)
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(x='sqft', y='price', hue='city', data=df, alpha=0.7, palette='viridis')
    plt.title('Price vs Square Footage by City')
    plt.xlabel('Square Footage')
    plt.ylabel('Price')
    scatter.yaxis.set_major_formatter(money_format)
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/price_vs_sqft_scatter.png', dpi=300)
    plt.close()
    
    # 3. Correlation heatmap for numerical features
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    heatmap = sns.heatmap(numeric_df.corr(), mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={'shrink': .8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Price by Property Type (boxplot)
    if 'homeType' in df.columns:
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(x='homeType', y='price', data=df, palette='Set2')
        plt.title('House Price by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Price')
        ax.yaxis.set_major_formatter(money_format)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/price_by_property_type.png', dpi=300)
        plt.close()
    
    # 5. Distribution of bedrooms and bathrooms (histogram)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['bed'].dropna(), kde=True, ax=axes[0], color='skyblue', bins=range(1, int(df['bed'].max()) + 2))
    axes[0].set_title('Distribution of Bedrooms')
    axes[0].set_xlabel('Number of Bedrooms')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(range(1, int(df['bed'].max()) + 1))
    
    sns.histplot(df['bath'].dropna(), kde=True, ax=axes[1], color='lightgreen', bins=np.arange(1, df['bath'].max() + 0.5, 0.5))
    axes[1].set_title('Distribution of Bathrooms')
    axes[1].set_xlabel('Number of Bathrooms')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('visualizations/bed_bath_distribution.png', dpi=300)
    plt.close()
    
    # 6. Price distribution (histogram)
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['price'], kde=True, bins=30, color='purple', alpha=0.7)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    ax.xaxis.set_major_formatter(money_format)
    plt.tight_layout()
    plt.savefig('visualizations/price_distribution.png', dpi=300)
    plt.close()
    
    # 7. Average price by number of bedrooms (bar plot)
    plt.figure(figsize=(12, 6))
    bed_price = df.groupby('bed')['price'].mean().reset_index()
    ax = sns.barplot(x='bed', y='price', data=bed_price, palette='Blues_d')
    plt.title('Average Price by Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price')
    ax.yaxis.set_major_formatter(money_format)
    for i, row in enumerate(bed_price.itertuples()):
        ax.text(i, row.price/2, f'${row.price:,.0f}', ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/avg_price_by_bedrooms.png', dpi=300)
    plt.close()
    
    # 8. City distribution (pie chart)
    plt.figure(figsize=(10, 8))
    city_counts = df['city'].value_counts()
    plt.pie(city_counts, labels=city_counts.index, autopct='%1.1f%%', startangle=90, 
            colors=sns.color_palette('pastel', len(city_counts)))
    plt.title('Distribution of Properties by City')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/city_distribution_pie.png', dpi=300)
    plt.close()
    
    # 9. Price per square foot by city (violin plot)
    if 'pricePerSf' in df.columns:
        plt.figure(figsize=(12, 7))
        ax = sns.violinplot(x='city', y='pricePerSf', data=df, palette='Set3')
        plt.title('Price per Square Foot by City')
        plt.xlabel('City')
        plt.ylabel('Price per Square Foot ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/price_per_sqft_violin.png', dpi=300)
        plt.close()
    
    # 10. Pairplot of numerical features
    # Select a subset of numerical columns to avoid too large of a plot
    numeric_cols = ['price', 'bed', 'bath', 'sqft']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(numeric_cols) >= 3:  # Need at least 3 columns for a meaningful pairplot
        pairplot_df = df[numeric_cols + ['city']].copy()
        # Limit the number of rows for performance
        pairplot_df = pairplot_df.sample(min(1000, len(pairplot_df)), random_state=42)
        
        pairplot = sns.pairplot(pairplot_df, hue='city', palette='viridis', 
                               plot_kws={'alpha': 0.6}, diag_kind='kde')
        plt.suptitle('Pairplot of Key Numerical Features', y=1.02, fontsize=16)
        plt.tight_layout()
        pairplot.savefig('visualizations/features_pairplot.png', dpi=300)
        plt.close('all')
    
    # 11. Bed to Bath Ratio Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['bed_bath_ratio'].dropna(), kde=True, bins=20, color='teal', alpha=0.7)
    plt.title('Distribution of Bed to Bath Ratio')
    plt.xlabel('Bed/Bath Ratio')
    plt.ylabel('Count')
    plt.axvline(1, color='red', linestyle='--', alpha=0.7, label='Equal Beds & Baths')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/bed_bath_ratio_distribution.png', dpi=300)
    plt.close()
    
    # 12. Price Trends by Lot Area (if available)
    if 'lotArea' in df.columns and df['lotArea'].notna().sum() > 10:
        plt.figure(figsize=(12, 6))
        # Remove extreme outliers for better visualization
        lotarea_df = df[(df['lotArea'] > 0) & (df['lotArea'] < df['lotArea'].quantile(0.99))].copy()
        sns.scatterplot(x='lotArea', y='price', hue='city', data=lotarea_df, alpha=0.6, palette='viridis')
        plt.title('Price vs Lot Area')
        plt.xlabel('Lot Area')
        plt.ylabel('Price')
        plt.gca().yaxis.set_major_formatter(money_format)
        plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('visualizations/price_vs_lotarea.png', dpi=300)
        plt.close()
    
    print(f"12 visualizations saved to the 'visualizations' directory")
    return

def create_advanced_visualizations(df):
    """
    Generate advanced data analysis visualizations that provide deeper insights
    into property price patterns and relationships between features.
    """
    
    
    
    # Create directory for advanced visualizations
    advanced_viz_dir = 'visualizations/advanced'
    os.makedirs(advanced_viz_dir, exist_ok=True)
    print("Generating advanced visualizations...")
    
    # Define a money formatter for price axes
    def money_formatter(x, pos):
        if x >= 1e6:
            return f'${x/1e6:.1f}M'
        else:
            return f'${x/1e3:.0f}K'
    
    money_format = FuncFormatter(money_formatter)
    
    # 1. Price to Square Foot Ratio by City (Boxplot)
    if 'sqft' in df.columns and df['sqft'].notna().sum() > 10:
        # Create a price/sqft column if not already present
        if 'pricePerSf' not in df.columns:
            df_with_ratio = df[df['sqft'] > 0].copy()
            df_with_ratio['pricePerSf'] = df_with_ratio['price'] / df_with_ratio['sqft']
        else:
            df_with_ratio = df[df['pricePerSf'].notna()].copy()
        
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='city', y='pricePerSf', data=df_with_ratio, palette='coolwarm')
        plt.title('Price per Square Foot Analysis by City')
        plt.xlabel('City')
        plt.ylabel('Price per Square Foot ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/price_per_sqft_by_city.png', dpi=300)
        plt.close()
    
    # 2. Heatmap of Price by Bedroom and Bathroom Count
    if all(col in df.columns for col in ['bed', 'bath']):
        # Create a pivot table of average prices
        bed_bath_pivot = df.pivot_table(
            values='price', 
            index='bed', 
            columns='bath', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(bed_bath_pivot, annot=True, fmt=',.0f', cmap='YlGnBu', linewidths=0.5)
        plt.title('Average Price by Bedroom and Bathroom Count')
        plt.xlabel('Number of Bathrooms')
        plt.ylabel('Number of Bedrooms')
        
        # Format annotations to show prices in $K
        for text in ax.texts:
            price_value = float(text.get_text().replace(',', ''))
            text.set_text(f'${price_value/1000:.0f}K')
        
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/price_bed_bath_heatmap.png', dpi=300)
        plt.close()
    
    # 3. Property Distribution by Type and City
    if 'homeType' in df.columns:
        plt.figure(figsize=(14, 8))
        property_type_by_city = pd.crosstab(df['city'], df['homeType'])
        property_type_by_city_percent = property_type_by_city.div(property_type_by_city.sum(axis=1), axis=0)
        
        ax = property_type_by_city_percent.plot(kind='bar', stacked=True, colormap='tab10')
        plt.title('Property Type Distribution by City')
        plt.xlabel('City')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, labels=[f'{x:.0%}' if x > 0.05 else '' for x in container])
        
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/property_type_by_city.png', dpi=300)
        plt.close()
    
    # 4. Price Range Distribution by City
    # Create price range categories
    price_ranges = [
        (0, 500000, '< $500K'),
        (500000, 1000000, '$500K-$1M'),
        (1000000, 1500000, '$1M-$1.5M'),
        (1500000, 2000000, '$1.5M-$2M'),
        (2000000, 3000000, '$2M-$3M'),
        (3000000, float('inf'), '$3M+')
    ]
    
    df_with_range = df.copy()
    df_with_range['price_range'] = pd.cut(
        df_with_range['price'],
        bins=[r[0] for r in price_ranges] + [float('inf')],
        labels=[r[2] for r in price_ranges],
        right=False
    )
    
    plt.figure(figsize=(14, 8))
    price_range_count = pd.crosstab(df_with_range['city'], df_with_range['price_range'])
    price_range_percent = price_range_count.div(price_range_count.sum(axis=1), axis=0)
    
    ax = price_range_percent.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Price Range Distribution by City')
    plt.xlabel('City')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{x:.0%}' if x > 0.05 else '' for x in container])
    
    plt.tight_layout()
    plt.savefig(f'{advanced_viz_dir}/price_range_by_city.png', dpi=300)
    plt.close()
    
    # 5. Price Trends by Square Footage and Bedrooms
    if 'sqft' in df.columns and df['sqft'].notna().sum() > 10:
        plt.figure(figsize=(14, 8))
        # Filter data to remove extreme outliers
        filtered_df = df[
            (df['sqft'] > 500) & 
            (df['sqft'] < df['sqft'].quantile(0.99)) & 
            (df['price'] > 100000) & 
            (df['price'] < df['price'].quantile(0.99))
        ].copy()
        
        # Create a categorical variable for bedrooms (limit to common values)
        filtered_df['bed_category'] = filtered_df['bed'].apply(
            lambda x: str(int(x)) if 1 <= x <= 5 else '6+' if x > 5 else 'Other'
        )
        
        # Create scatter plot
        scatter = sns.scatterplot(
            data=filtered_df,
            x='sqft',
            y='price',
            hue='bed_category',
            palette='viridis',
            alpha=0.7,
            s=80
        )
        
        # Add regression lines for each bedroom category
        for bed_cat, color in zip(sorted(filtered_df['bed_category'].unique()), 
                                 sns.color_palette('viridis', len(filtered_df['bed_category'].unique()))):
            if bed_cat != 'Other':
                subset = filtered_df[filtered_df['bed_category'] == bed_cat]
                if len(subset) > 5:  # Only add line if we have enough data points
                    sns.regplot(
                        x='sqft',
                        y='price',
                        data=subset,
                        scatter=False,
                        ci=None,
                        line_kws={'linestyle': '--', 'linewidth': 1},
                        color=color,
                        ax=scatter
                    )
        
        plt.title('Price vs. Square Footage by Number of Bedrooms')
        plt.xlabel('Square Footage')
        plt.ylabel('Price')
        scatter.yaxis.set_major_formatter(money_format)
        plt.legend(title='Bedrooms', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/price_sqft_bedroom_trends.png', dpi=300)
        plt.close()
    
    # 6. Geographic Pricing Bubble Chart (if coordinates available)
    if all(col in df.columns for col in ['latitude', 'longitude']):
        plt.figure(figsize=(14, 10))
        
        # Create a bubble chart
        scatter = plt.scatter(
            df['longitude'],
            df['latitude'],
            s=df['price'] / 50000,  # Size proportional to price
            c=df['price'],  # Color by price
            cmap='plasma',
            alpha=0.6,
            edgecolors='w',
            linewidths=0.5
        )
        
        plt.title('Geographic Price Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Price')
        cbar.ax.yaxis.set_major_formatter(money_format)
        
        # Add city labels
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            plt.annotate(
                city,
                xy=(city_data['longitude'].mean(), city_data['latitude'].mean()),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8),
                fontsize=12,
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/geographic_price_bubble.png', dpi=300)
        plt.close()
    
    # 7. Price to Lot Area Ratio Analysis
    if 'lotArea' in df.columns and df['lotArea'].notna().sum() > 10:
        # Create a filtered dataframe without extreme outliers
        lot_df = df[
            (df['lotArea'] > 0) & 
            (df['lotArea'] < df['lotArea'].quantile(0.99)) & 
            (df['price'] > 100000)
        ].copy()
        
        # Calculate price per lot area
        lot_df['pricePerLotArea'] = lot_df['price'] / lot_df['lotArea']
        
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='city', y='pricePerLotArea', data=lot_df, palette='Spectral')
        plt.title('Price per Lot Area Unit by City')
        plt.xlabel('City')
        plt.ylabel('Price per Lot Area Unit ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/price_per_lot_area.png', dpi=300)
        plt.close()
    
    # 8. Density Plot of Price by City
    plt.figure(figsize=(14, 8))
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        sns.kdeplot(city_data['price'], label=city, fill=True, alpha=0.3)
    
    plt.title('Price Density Distribution by City')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.gca().xaxis.set_major_formatter(money_format)
    plt.legend(title='City')
    plt.tight_layout()
    plt.savefig(f'{advanced_viz_dir}/price_density_by_city.png', dpi=300)
    plt.close()
    
    # 9. Bed-Bath Ratio vs Price Analysis
    plt.figure(figsize=(12, 8))
    ratio_df = df.dropna(subset=['bed', 'bath', 'price']).copy()
    
    # Remove extreme outliers for better visualization
    ratio_df = ratio_df[
        (ratio_df['bed_bath_ratio'] > 0) &
        (ratio_df['bed_bath_ratio'] < 5) &
        (ratio_df['price'] < ratio_df['price'].quantile(0.99))
    ]
    
    sns.scatterplot(
        x='bed_bath_ratio',
        y='price',
        hue='city',
        palette='Set1',
        data=ratio_df,
        alpha=0.6
    )
    
    plt.title('Price vs Bed-Bath Ratio')
    plt.xlabel('Bed to Bath Ratio')
    plt.ylabel('Price')
    plt.gca().yaxis.set_major_formatter(money_format)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Equal Beds & Baths')
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{advanced_viz_dir}/price_vs_bed_bath_ratio.png', dpi=300)
    plt.close()
    
    # 10. Price Forecasting by Area (Binned Square Footage)
    if 'sqft' in df.columns and df['sqft'].notna().sum() > 20:
        price_forecast_df = df.dropna(subset=['sqft', 'price']).copy()
        
        # Create square footage bins
        sqft_bins = [0, 1000, 1500, 2000, 2500, 3000, 4000, float('inf')]
        sqft_labels = ['<1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-4000', '4000+']
        
        price_forecast_df['sqft_range'] = pd.cut(
            price_forecast_df['sqft'],
            bins=sqft_bins,
            labels=sqft_labels,
            right=False
        )
        
        # Calculate average price by city and square footage range
        price_by_sqft_city = price_forecast_df.pivot_table(
            values='price',
            index='city',
            columns='sqft_range',
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(
            price_by_sqft_city,
            annot=True,
            fmt=',.0f',
            cmap='YlOrRd',
            linewidths=0.5
        )
        
        plt.title('Average Price by City and Square Footage Range')
        plt.xlabel('Square Footage Range')
        plt.ylabel('City')
        
        # Format annotations to show prices in $K or $M
        for text in ax.texts:
            price_value = float(text.get_text().replace(',', ''))
            if price_value > 0:  # Only format non-zero values
                if price_value >= 1e6:
                    text.set_text(f'${price_value/1e6:.1f}M')
                else:
                    text.set_text(f'${price_value/1000:.0f}K')
            else:
                text.set_text('')
        
        plt.tight_layout()
        plt.savefig(f'{advanced_viz_dir}/price_forecast_by_area.png', dpi=300)
        plt.close()
    
    print(f"10 advanced visualizations saved to '{advanced_viz_dir}' directory")
    return

def main():
    # Define paths to the raw and processed data directories
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')
    
    try:
        # Combine individual CSV files into one DataFrame
        combined_df = combine_csv_files(raw_data_path, processed_data_path)
        
        # Load and preprocess the combined data
        processed_df, features, target = load_data(os.path.join(processed_data_path, 'combined_properties.csv'))
        
        # Generate visualizations
        visualize_data(processed_df)
        
    except FileNotFoundError as e:
        print(f"Error: One or more CSV files not found. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()