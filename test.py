import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(y=[4, 2, 7]))
fig.write_image("success_plot.png")
print("Saved successfully!")
