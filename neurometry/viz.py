# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

#Returns a custom donut figure
def donut(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+value+percent')])
    fig.update_layout(title_text=title)
    return fig

#Returns a custom histogram figure
def histogram(x, title):
    fig = go.Figure(data=[go.Histogram(x=x, name="count", texttemplate="%{y}", textfont_size=20)])
    fig.update_layout(title_text=title, template="plotly_white")
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    return fig

#Returns a scatter plot with trendline fitted via OLS
def scatter(data, x, y, title):
    fig = px.scatter(data, x, y, trendline="ols", trendline_color_override='pink', title=title)
    return fig

#Returns a heatmap from a groupby object
def heatmap(data, xlist, ylist, xlabel, ylabel, zlabel):
    fig = px.imshow(data.iloc[:,1:],
                labels=dict(x=xlabel, y=ylabel, color=zlabel),
                y=ylist,
                x=xlist,
                color_continuous_scale='RdBu_r', aspect='auto'
               )
    fig.update_xaxes(side="top")
    fig.update_layout(xaxis_nticks=len(xlist), yaxis_nticks=len(ylist))
    return fig

#Returns sorted heatmap with user defined clusters
def nt_heatmap(dataset, cluster):
    #dataset["id"] = dataset["id"].apply(str)
    dataset = dataset[["id","gaba", "ach", "glut", "oct", "ser", "da"]]
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(dataset[["gaba", "ach", "glut", "oct", "ser", "da"]])
    dataset["labels"] = list(kmeans.labels_)
    _sorted = dataset.sort_values(by=["labels"])

    fig = go.Figure(data=go.Heatmap(
                    z=_sorted[["gaba", "ach", "glut", "oct", "ser", "da", "labels"]].iloc[:,:-1], x=["gaba", "ach", "glut", "oct", "ser", "da"],
                    colorscale='RdBu_r'))
    fig.update_xaxes(side="top")
    fig.update_layout(title="NT Confidence Scores")
    return fig, _sorted

#Returns a bar chart
def bar(x, y, title):
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text=title, xaxis_tickangle=-45,
    template="plotly_white")
    return fig


# %%
