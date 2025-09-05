import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from datetime import datetime, timedelta
import textwrap

def create_synthetic_plant_disease_data():
    """Create synthetic plant disease dataset similar to the Kaggle dataset structure"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define plant types and diseases (common in plant disease datasets)
    plants = ['Tomato', 'Potato', 'Corn', 'Apple', 'Grape', 'Pepper', 'Strawberry', 'Peach', 'Cherry', 'Soybean']
    
    diseases = {
        'Tomato': ['Early_blight', 'Late_blight', 'Leaf_mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_spot', 'Mosaic_virus', 'Yellow_leaf_curl_virus', 'Bacterial_spot', 'Healthy'],
        'Potato': ['Early_blight', 'Late_blight', 'Healthy'],
        'Corn': ['Cercospora_leaf_spot', 'Common_rust', 'Northern_leaf_blight', 'Healthy'],
        'Apple': ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'Healthy'],
        'Grape': ['Black_rot', 'Esca', 'Leaf_blight', 'Healthy'],
        'Pepper': ['Bacterial_spot', 'Healthy'],
        'Strawberry': ['Leaf_scorch', 'Healthy'],
        'Peach': ['Bacterial_spot', 'Healthy'],
        'Cherry': ['Powdery_mildew', 'Healthy'],
        'Soybean': ['Healthy']
    }
    
    # Generate synthetic data
    n_samples = 2500  # Realistic dataset size
    
    data = []
    
    for i in range(n_samples):
        plant = random.choice(plants)
        disease = random.choice(diseases[plant])
        
        # Generate realistic image path
        image_path = f"PlantVillage/{plant}___{disease}/{plant}_{disease}_{random.randint(1000, 9999)}.jpg"
        
        # Generate some numeric features that might be in a plant disease dataset
        leaf_area = round(np.random.normal(25.5, 8.2), 2)  # cm²
        lesion_count = np.random.poisson(3) if disease != 'Healthy' else 0
        disease_severity = round(np.random.uniform(0, 100), 1) if disease != 'Healthy' else 0.0
        humidity = round(np.random.normal(65, 15), 1)
        temperature = round(np.random.normal(24.5, 6.8), 1)
        ph_level = round(np.random.normal(6.5, 0.8), 2)
        
        # Image properties
        image_width = random.choice([224, 256, 512])
        image_height = image_width  # Assuming square images
        color_channels = 3
        
        # Classification confidence (if this was model output)
        confidence = round(np.random.uniform(0.75, 0.99), 4) if disease != 'Healthy' else round(np.random.uniform(0.85, 0.99), 4)
        
        data.append({
            'image_path': image_path,
            'plant_type': plant,
            'disease_class': f"{plant}___{disease}",
            'disease_name': disease.replace('_', ' ').title(),
            'is_healthy': disease == 'Healthy',
            'leaf_area_cm2': leaf_area,
            'lesion_count': lesion_count,
            'disease_severity_pct': disease_severity,
            'humidity_pct': humidity,
            'temperature_celsius': temperature,
            'soil_ph': ph_level,
            'image_width': image_width,
            'image_height': image_height,
            'channels': color_channels,
            'confidence_score': confidence,
            'capture_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'dataset_split': random.choices(['train', 'validation', 'test'], weights=[0.7, 0.15, 0.15])[0]
        })
    
    return pd.DataFrame(data)

def wrap_text(text, width=15):
    """Wrap text to specified width"""
    if pd.isna(text):
        return text
    text = str(text)
    if len(text) <= width:
        return text
    return '\n'.join(textwrap.wrap(text, width=width))

def format_table_data(df, max_width=12):
    """Format dataframe for better table display with text wrapping"""
    formatted_df = df.copy()
    
    for col in formatted_df.columns:
        # Special handling for different column types
        if col == 'image_path':
            # Show only filename for image paths
            formatted_df[col] = formatted_df[col].str.split('/').str[-1]
            formatted_df[col] = formatted_df[col].apply(lambda x: wrap_text(x, 20))
        elif col == 'disease_class':
            # Shorten disease class names
            formatted_df[col] = formatted_df[col].str.replace('___', '_')
            formatted_df[col] = formatted_df[col].apply(lambda x: wrap_text(x, 15))
        elif formatted_df[col].dtype == 'object':
            # Wrap other text columns
            formatted_df[col] = formatted_df[col].apply(lambda x: wrap_text(x, max_width))
        elif formatted_df[col].dtype in ['float64', 'int64']:
            # Format numbers to reasonable precision
            if formatted_df[col].dtype == 'float64':
                formatted_df[col] = formatted_df[col].round(2)
    
    return formatted_df

def create_dataset_screenshots():
    # Set up the style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("Generating synthetic plant disease dataset...")
    df = create_synthetic_plant_disease_data()
    print(f"Synthetic dataset created successfully. Shape: {df.shape}")
    
    # Create dataset overview with better layout
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Plant Disease Dataset - Overview', fontsize=20, fontweight='bold', y=0.95)
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.2)
    
    # Dataset info summary (top, spanning both columns)
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')
    
    info_text = f"""Dataset Information:
    
Shape: {df.shape[0]} rows × {df.shape[1]} columns  |  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB  |  Missing values: {df.isnull().sum().sum()}

Plant Types: {', '.join(df['plant_type'].unique())}
Disease Classes: {df['disease_class'].nunique()} unique classes
Healthy vs Diseased: {df['is_healthy'].sum()} healthy, {len(df) - df['is_healthy'].sum()} diseased samples
"""
    
    ax_info.text(0.05, 0.8, info_text, transform=ax_info.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", alpha=0.9, edgecolor="#2196F3"))
    
    # Sample data display - showing key columns only
    key_columns = ['plant_type', 'disease_name', 'is_healthy', 'lesion_count', 'disease_severity_pct', 'confidence_score']
    
    # First 5 rows (left)
    ax_head = fig.add_subplot(gs[1, 0])
    ax_head.axis('off')
    ax_head.set_title('First 5 Rows (Sample)', fontsize=14, fontweight='bold', pad=20, color='#2E7D32')
    
    head_data = df[key_columns].head().reset_index(drop=True)
    # Add row index
    head_data.insert(0, 'Row', range(1, len(head_data) + 1))
    
    # Format data with text wrapping
    formatted_head = format_table_data(head_data, max_width=10)
    
    # Create clean table
    table1 = ax_head.table(cellText=formatted_head.values,
                          colLabels=[wrap_text(col, 8) for col in formatted_head.columns],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the table
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1.2, 3)  # Increased height for wrapped text
    
    # Header styling
    for i in range(len(formatted_head.columns)):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
        table1[(0, i)].set_height(0.2)
    
    # Row styling with dynamic height for wrapped text
    for i in range(1, len(formatted_head) + 1):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(len(formatted_head.columns)):
            table1[(i, j)].set_facecolor(color)
            # Check if cell contains newlines (wrapped text)
            cell_text = str(formatted_head.iloc[i-1, j])
            lines = cell_text.count('\n') + 1
            height = max(0.15, 0.05 * lines)
            table1[(i, j)].set_height(height)
    
    # Random 5 rows (right)
    ax_random = fig.add_subplot(gs[1, 1])
    ax_random.axis('off')
    ax_random.set_title('5 Random Rows (Sample)', fontsize=14, fontweight='bold', pad=20, color='#1976D2')
    
    random_data = df[key_columns].sample(n=5, random_state=42).reset_index(drop=True)
    # Add row index
    original_indices = df.sample(n=5, random_state=42).index.tolist()
    random_data.insert(0, 'Row', [f"{i+1}" for i in original_indices])
    
    # Format data with text wrapping
    formatted_random = format_table_data(random_data, max_width=10)
    
    table2 = ax_random.table(cellText=formatted_random.values,
                            colLabels=[wrap_text(col, 8) for col in formatted_random.columns],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    
    # Style the table
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.2, 3)  # Increased height for wrapped text
    
    # Header styling
    for i in range(len(formatted_random.columns)):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')
        table2[(0, i)].set_height(0.2)
    
    # Row styling with dynamic height for wrapped text
    for i in range(1, len(formatted_random) + 1):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(len(formatted_random.columns)):
            table2[(i, j)].set_facecolor(color)
            # Check if cell contains newlines (wrapped text)
            cell_text = str(formatted_random.iloc[i-1, j])
            lines = cell_text.count('\n') + 1
            height = max(0.15, 0.05 * lines)
            table2[(i, j)].set_height(height)
    
    # Statistics summary (bottom, spanning both columns)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    ax_stats.set_title('Dataset Statistics & Column Information', fontsize=14, fontweight='bold', pad=20, color='#6A1B9A')
    
    # Create statistics summary
    stats_data = []
    
    # Add column information
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        if df[col].dtype in ['int64', 'float64']:
            stats_info = f"μ={df[col].mean():.1f}, σ={df[col].std():.1f}"
        else:
            top_value = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'
            stats_info = f"Top: {wrap_text(str(top_value), 15)}"
        
        stats_data.append([wrap_text(col, 12), dtype, unique_count, null_count, stats_info])
    
    # Show first 8 columns in a table
    stats_df = pd.DataFrame(stats_data[:8], 
                           columns=['Column', 'Data Type', 'Unique\nValues', 'Null\nCount', 'Statistics'])
    
    table3 = ax_stats.table(cellText=stats_df.values,
                           colLabels=stats_df.columns,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0.2, 1, 0.6])
    
    table3.auto_set_font_size(False)
    table3.set_fontsize(8)
    table3.scale(1.2, 2.2)  # Increased height for wrapped text
    
    # Header styling
    for i in range(len(stats_df.columns)):
        table3[(0, i)].set_facecolor('#9C27B0')
        table3[(0, i)].set_text_props(weight='bold', color='white')
        table3[(0, i)].set_height(0.15)
    
    # Row styling with dynamic height
    for i in range(1, len(stats_df) + 1):
        color = '#fce4ec' if i % 2 == 0 else 'white'
        for j in range(len(stats_df.columns)):
            table3[(i, j)].set_facecolor(color)
            # Check if cell contains newlines (wrapped text)
            cell_text = str(stats_df.iloc[i-1, j])
            lines = cell_text.count('\n') + 1
            height = max(0.12, 0.04 * lines)
            table3[(i, j)].set_height(height)
    
    # Add remaining columns info as text
    if len(df.columns) > 8:
        remaining_text = f"... and {len(df.columns) - 8} more columns: " + ", ".join(df.columns[8:])
        ax_stats.text(0.05, 0.1, remaining_text, transform=ax_stats.transAxes, 
                     fontsize=10, style='italic', color='#666')
    
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Dataset overview screenshot saved as 'dataset_overview.png'")
    
    # Create a clean, professional table for the 5 random rows
    fig2 = plt.figure(figsize=(18, 10))
    ax = fig2.add_subplot(111)
    ax.axis('off')
    
    # Title
    fig2.suptitle('Plant Disease Dataset - Random Sample Data', 
                 fontsize=18, fontweight='bold', y=0.92)
    
    # Get all columns for detailed view but limit to most important ones
    detailed_columns = ['image_path', 'plant_type', 'disease_name', 'is_healthy', 
                       'lesion_count', 'disease_severity_pct', 'humidity_pct', 
                       'temperature_celsius', 'confidence_score', 'dataset_split']
    
    random_detailed = df[detailed_columns].sample(n=5, random_state=42).reset_index(drop=True)
    
    # Add a sample ID column
    random_detailed.insert(0, 'Sample_ID', [f"S{i+1:02d}" for i in random_detailed.index])
    
    # Format the data with text wrapping
    display_data = format_table_data(random_detailed, max_width=12)
    
    # Wrap column headers
    wrapped_columns = [wrap_text(col.replace('_', ' ').title(), 8) for col in display_data.columns]
    
    # Create the table
    table = ax.table(cellText=display_data.values,
                    colLabels=wrapped_columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0.15, 1, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Increased from 7 to 10 for better readability
    table.scale(1, 3.8)  # Slightly increased height for larger text
    
    # Professional styling
    # Header row
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#1565C0')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.12)
    
    # Data rows with alternating colors and dynamic heights
    colors = ['#E3F2FD', '#FFFFFF']
    for i in range(1, len(display_data) + 1):
        color = colors[i % 2]
        for j in range(len(display_data.columns)):
            table[(i, j)].set_facecolor(color)
            
            # Calculate height based on text content
            cell_text = str(display_data.iloc[i-1, j])
            lines = cell_text.count('\n') + 1
            height = max(0.08, 0.03 * lines)
            table[(i, j)].set_height(height)
            
            # Highlight healthy vs diseased
            if 'is_healthy' in display_data.columns and j == display_data.columns.get_loc('is_healthy'):
                if display_data.iloc[i-1]['is_healthy']:
                    table[(i, j)].set_facecolor('#C8E6C9')
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')
    
    # Add subtitle
    subtitle = f"Showing 5 randomly selected samples from {len(df)} total records"
    ax.text(0.5, 0.08, subtitle, transform=ax.transAxes, ha='center', 
           fontsize=12, style='italic', color='#666')
    
    plt.savefig('random_rows_detailed.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Detailed random rows screenshot saved as 'random_rows_detailed.png'")
    
    plt.show()
    
    print(f"\n🎉 Screenshots generated successfully!")
    print(f"📊 Dataset used: Synthetic Plant Disease Dataset")
    print(f"📋 Dataset shape: {df.shape}")
    print("\nFiles created:")
    print("• dataset_overview.png - Complete overview with dataset info")
    print("• random_rows_detailed.png - Detailed view of 5 random rows")
    
if __name__ == "__main__":
    create_dataset_screenshots()