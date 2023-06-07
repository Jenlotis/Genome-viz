import plotly.graph_objects as go

seq = "ABCDAABDCC"
main_colors = {'A': '#001BFF', 'B': '#FF0000', 'C': '#FFF000', 'D': '#2FCF00'}
fill_colors = {'A': '#5A6CFF', 'B': '#FF5D5D', 'C': '#FFF557', 'D': '#6ECE52'}

fig = go.Figure(data=go.Scatter())

for i, letter in enumerate(seq):
    fig.add_annotation(
        x=i+1, y=3,  # Coordinates where the text should be placed
        text=f'<b>{letter}</b>',  # The text to be displayed
        showarrow=False,  # Do not show an arrow pointing to the annotation
        font=dict(
            family="Arial",  # Font family
            size=16,  # Font size
            color='black',
        ),
        bordercolor='rgba(0, 0, 0, 0)',
        borderwidth=2,
        borderpad=24,
        bgcolor=fill_colors[letter]
    )

fig.update_layout(
    plot_bgcolor="white",  # Set the background color to white
    width=800,
    height=400,
    xaxis=dict(
        showgrid=False,  # Hide the x-axis grid
        showticklabels=False
    ),
    yaxis=dict(
        showgrid=False,  # Hide the y-axis grid
        showticklabels=False
    )
)

fig.show()
