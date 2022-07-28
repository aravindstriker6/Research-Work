import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def HeatMap(data,annot,xticklabels,yticklabels,figsize_x,figsize_y):
    plt.figure(figsize=(figsize_x,figsize_y))
    hm = sn.heatmap(data=data,square=True,
                annot=annot, xticklabels=xticklabels,
                yticklabels=yticklabels)
    plt.show()
    return hm

def Scatterplot(x,y,names,tickvals_y,tickvals_x,title,xaxis_title,yaxis_title,legend_title,font):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=x,
    y=y,
    name=names))
    fig.update_yaxes(tickvals=tickvals_y,ticks="outside", tickwidth=2, tickcolor='blue', ticklen=6)
    fig.update_xaxes(tickvals=tickvals_x,ticks="outside", tickwidth=2, tickcolor='blue', ticklen=6)
    fig.update_layout(
    title=title,
    xaxis_title=xaxis_title,
    yaxis_title=yaxis_title,
    legend_title=legend_title,
    font=font)
    fig.show()
    return fig
