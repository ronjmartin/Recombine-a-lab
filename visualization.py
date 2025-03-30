import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

def create_visualization(df: pd.DataFrame, x_col: str, y_col: str, viz_type: str):
    """
    Create a visualization based on the given data and type
    
    Args:
        df (pd.DataFrame): Data to visualize
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        viz_type (str): Type of visualization
        
    Returns:
        plotly.graph_objects.Figure: Visualization figure
    """
    # Convert data to numeric if possible
    try:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    except:
        pass
    
    try:
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    except:
        pass
    
    # Create visualization based on type
    if viz_type == "Bar Chart":
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col,
            title=f"{y_col} by {x_col}",
            labels={x_col: x_col, y_col: y_col}
        )
    elif viz_type == "Line Chart":
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=f"{y_col} over {x_col}",
            labels={x_col: x_col, y_col: y_col}
        )
    elif viz_type == "Scatter Plot":
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            title=f"{y_col} vs {x_col}",
            labels={x_col: x_col, y_col: y_col}
        )
    else:
        # Default to bar chart
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col,
            title=f"{y_col} by {x_col}",
            labels={x_col: x_col, y_col: y_col}
        )
    
    # Update layout for better readability
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_sentiment_gauge(sentiment_score: float):
    """
    Create a gauge chart for sentiment score
    
    Args:
        sentiment_score (float): Sentiment score (-1 to 1)
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    # Map sentiment score from [-1, 1] to [0, 1]
    normalized_score = (sentiment_score + 1) / 2
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': 'red'},
                {'range': [-0.5, 0], 'color': 'salmon'},
                {'range': [0, 0.5], 'color': 'lightblue'},
                {'range': [0.5, 1], 'color': 'green'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_emotion_chart(emotion_df: pd.DataFrame):
    """
    Create a bar chart for emotions
    
    Args:
        emotion_df (pd.DataFrame): DataFrame with emotions and scores
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Create horizontal bar chart
    fig = px.bar(
        emotion_df,
        x="Score",
        y="Emotion",
        orientation='h',
        title="Emotion Distribution",
        color="Score",
        color_continuous_scale=["red", "yellow", "green"]
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_section_sentiment_chart(section_df: pd.DataFrame):
    """
    Create a chart showing sentiment by document section
    
    Args:
        section_df (pd.DataFrame): DataFrame with section sentiments
        
    Returns:
        plotly.graph_objects.Figure: Chart figure
    """
    # Create line chart
    fig = px.line(
        section_df,
        x="Section",
        y="Score",
        title="Sentiment Flow Throughout Document",
        markers=True
    )
    
    # Add horizontal reference lines
    fig.add_shape(
        type="line",
        x0=section_df["Section"].iloc[0],
        y0=0,
        x1=section_df["Section"].iloc[-1],
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Color positive values green and negative values red
    for i, score in enumerate(section_df["Score"]):
        color = "green" if score >= 0 else "red"
        fig.add_trace(
            go.Scatter(
                x=[section_df["Section"].iloc[i]],
                y=[score],
                mode="markers",
                marker=dict(color=color, size=10),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_entity_chart(entities: List[Dict[str, str]]):
    """
    Create a visualization of named entities
    
    Args:
        entities (List[Dict[str, str]]): List of entity dictionaries
        
    Returns:
        plotly.graph_objects.Figure: Visualization figure
    """
    # Count entity types
    entity_types = {}
    for entity in entities:
        entity_type = entity['type']
        if entity_type in entity_types:
            entity_types[entity_type] += 1
        else:
            entity_types[entity_type] = 1
    
    # Create DataFrame for visualization
    entity_df = pd.DataFrame({
        'Type': list(entity_types.keys()),
        'Count': list(entity_types.values())
    })
    
    # Sort by count
    entity_df = entity_df.sort_values('Count', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        entity_df,
        x='Count',
        y='Type',
        orientation='h',
        title="Named Entity Types",
        color='Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig
