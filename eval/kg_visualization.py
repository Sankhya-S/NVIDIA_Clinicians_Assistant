import networkx as nx
import plotly.graph_objects as go
import json
import numpy as np
import os

def visualize_knowledge_graph(data_path=None, output_dir=None):
    """
    Visualize a medical knowledge graph with enhanced metadata and entity connections.
    
    Args:
        data_path (str): Path to the knowledge graph JSON file
        output_dir (str): Directory to save the visualization
        
    Returns:
        dict: Metrics about the visualization including node and edge counts
    """
    # Set default paths if none provided
    if data_path is None:
        data_path = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/output/knowledge_graph.json"
    
    if output_dir is None:
        output_dir = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/output/"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'knowledge_graph_detailed.html')
    
    # Load and validate knowledge graph data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge graph file not found at {data_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in knowledge graph file")
        return None
    
    # Initialize directed graph
    G = nx.DiGraph()
    
    # Define color scheme for different node types
    colors = {
        "document": "#4CAF50",       # Green for documents
        "entity": "#2196F3",         # Blue for entities
        "theme": "#FFC107",          # Yellow for themes
        "metadata": "#FF5722",       # Orange for metadata
        "relationship": "#9C27B0",   # Purple for relationships
        "default": "#757575"         # Gray for default
    }

    # Initialize node attribute lists
    node_colors = []
    node_sizes = []
    node_texts = []
    node_symbols = []
    added_nodes = set()
    
    # Process all relationships in the graph
    for rel in data.get('relationships', []):
        # Process source node
        source_id = rel['source']['id']
        if source_id not in added_nodes:
            source_properties = rel['source'].get('properties', {})
            
            # Add document node
            G.add_node(source_id)
            added_nodes.add(source_id)
            
            # Create detailed hover text for document
            hover_text = []
            hover_text.append(f"ID: {source_id}")
            
            # Add content preview
            if 'page_content' in source_properties:
                content_preview = source_properties['page_content'][:200] + "..."
                hover_text.append(f"Content: {content_preview}")
            
            # Add metadata information
            if 'document_metadata' in source_properties:
                metadata = source_properties['document_metadata']
                hover_text.append("Metadata:")
                for key, value in metadata.items():
                    if key != 'embedding':  # Skip embedding vectors in visualization
                        hover_text.append(f"- {key}: {value}")
            
            # Add themes and entities
            if 'themes' in source_properties:
                hover_text.append(f"Themes: {', '.join(source_properties['themes'])}")
            if 'entities' in source_properties:
                hover_text.append(f"Entities: {', '.join(source_properties['entities'])}")
            
            # Add node attributes
            node_texts.append('<br>'.join(hover_text))
            node_colors.append(colors['document'])
            node_sizes.append(30)
            node_symbols.append('circle')
            
            # Process and add entity nodes
            if 'entities' in source_properties:
                for entity in source_properties['entities']:
                    entity_id = f"entity_{entity}"
                    if entity_id not in added_nodes:
                        G.add_node(entity_id)
                        added_nodes.add(entity_id)
                        node_texts.append(f"Entity: {entity}")
                        node_colors.append(colors['entity'])
                        node_sizes.append(20)
                        node_symbols.append('diamond')
                    G.add_edge(source_id, entity_id)
            
            # Process and add theme nodes
            if 'themes' in source_properties:
                for theme in source_properties['themes']:
                    theme_id = f"theme_{theme}"
                    if theme_id not in added_nodes:
                        G.add_node(theme_id)
                        added_nodes.add(theme_id)
                        node_texts.append(f"Theme: {theme}")
                        node_colors.append(colors['theme'])
                        node_sizes.append(15)
                        node_symbols.append('square')
                    G.add_edge(source_id, theme_id)

        # Process target node if it exists
        if 'target' in rel:
            target_id = rel['target']['id']
            if target_id not in added_nodes:
                target_properties = rel['target'].get('properties', {})
                G.add_node(target_id)
                added_nodes.add(target_id)
                
                # Create hover text for target
                hover_text = []
                hover_text.append(f"ID: {target_id}")
                
                if 'page_content' in target_properties:
                    content_preview = target_properties['page_content'][:200] + "..."
                    hover_text.append(f"Content: {content_preview}")
                
                if 'document_metadata' in target_properties:
                    metadata = target_properties['document_metadata']
                    hover_text.append("Metadata:")
                    for key, value in metadata.items():
                        if key != 'embedding':
                            hover_text.append(f"- {key}: {value}")
                
                node_texts.append('<br>'.join(hover_text))
                node_colors.append(colors['document'])
                node_sizes.append(30)
                node_symbols.append('circle')

            # Add edge with properties
            edge_properties = rel.get('properties', {})
            G.add_edge(source_id, target_id, **edge_properties)

    # Calculate layout with adjusted spacing
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add edge properties to hover text
        if edge[2]:
            edge_texts.extend([str(edge[2]), None, None])
        else:
            edge_texts.extend([None, None, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_texts,
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_texts,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line_width=2))

    # Create and configure figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Medical Knowledge Graph with Metadata',
                       titlefont_size=16,
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[
                           dict(
                               text="Hover over nodes and edges to see details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0, y=-0.1
                           )
                       ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    # Save visualization
    try:
        fig.write_html(output_file)
        print(f"Visualization saved to: {output_file}")
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
        return None
    
    # Return metrics about the visualization
    metrics = {
        'num_nodes': len(G.nodes()),
        'num_edges': len(G.edges()),
        'num_documents': len([n for n in G.nodes() if not (n.startswith('entity_') or n.startswith('theme_'))]),
        'num_entities': len([n for n in G.nodes() if n.startswith('entity_')]),
        'num_themes': len([n for n in G.nodes() if n.startswith('theme_')]),
        'output_file': output_file
    }
    
    return metrics

if __name__ == "__main__":
    # Run visualization with default paths
    metrics = visualize_knowledge_graph()
    
    if metrics:
        print("\nVisualization metrics:")
        for key, value in metrics.items():
            print(f"- {key}: {value}")
        
        print("\nNode types in visualization:")
        print("- Documents: shown as green circles")
        print("- Entities: shown as blue diamonds")
        print("- Themes: shown as yellow squares")
        print("\nInteraction instructions:")
        print("- Hover over nodes to see detailed information")
        print("- Hover over edges to see relationship properties")
        print("- Use mouse wheel to zoom in/out")
        print("- Click and drag to pan around the visualization")